extern crate crossbeam_epoch as epoch;

use std::borrow::Borrow;
use std::cmp;
use std::mem;
use std::ptr;
use std::slice;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst};

use epoch::{Atomic, Guard, Shared};

// TODO: a test where comparison function sometimes panics
// TODO: Explain why TrustedOrd is not required
// TODO: In remove directly relink instead of calling search_level?

// TODO: Make sure not to execute user defined code within an epoch!! not even Drop for K and V
// TODO: fn drain()
// TODO: fn extend()
// TODO: optimize when V is a ZST
// TODO: fn len() and fn is_empty()

// TODO: inspiration is RocksDB's skiplist
// TODO: another inspiration is ConcurrentSkipListMap (not necessarily as fast as possible)

const MAX_HEIGHT: usize = 1 << HEIGHT_BITS;
const HEIGHT_BITS: usize = 5;
const HEIGHT_MASK: usize = (1 << HEIGHT_BITS) - 1;

/// A skiplist node.
///
/// This struct is marked with `repr(C)` so that the specific order of fields can be enforced.
/// It is important that the tower is the last field since it is dynamically sized. The key,
/// reference count, and height are kept close to the tower to improve cache locality during
/// skiplist traversal.
#[repr(C)]
struct Node<K, V> {
    /// The value.
    value: V,

    /// The key.
    key: K,

    /// Keeps the number of references to the node and the height of its tower.
    refs_and_height: AtomicUsize,

    /// The tower of atomic pointers.
    pointers: [Atomic<Node<K, V>>; 1],
}

impl<K, V> Node<K, V> {
    /// Allocate a node.
    ///
    /// The returned node will start with reference count of zero and the tower will be initialized
    /// with null pointers. However, the key and the value will be left uninitialized, and that is
    /// why this function is unsafe.
    unsafe fn alloc(height: usize) -> *mut Self {
        let cap = Self::size_in_u64s(height);
        let mut v = Vec::<u64>::with_capacity(cap);
        let ptr = v.as_mut_ptr() as *mut Self;
        mem::forget(v);

        ptr::write(&mut (*ptr).refs_and_height, AtomicUsize::new(height - 1));
        ptr::write_bytes(&mut (*ptr).pointers[0], 0, height);
        ptr
    }

    /// Deallocate a node.
    ///
    /// This function will not run any destructors.
    unsafe fn dealloc(ptr: *mut Self) {
        let cap = Self::size_in_u64s((*ptr).height());
        drop(Vec::from_raw_parts(ptr as *mut u64, 0, cap));
    }

    /// Returns the size of a node with tower of given `height`, measured in `u64`s.
    fn size_in_u64s(height: usize) -> usize {
        assert!(1 <= height && height <= MAX_HEIGHT);
        assert!(mem::align_of::<Self>() <= mem::align_of::<u64>());

        let size_base = mem::size_of::<Self>();
        let size_ptr = mem::size_of::<Atomic<Self>>();

        let size_u64 = mem::size_of::<u64>();
        let size_self = size_base
            .checked_add(size_ptr.checked_mul(height - 1).unwrap())
            .unwrap();

        size_self.checked_add(size_u64 - 1).unwrap() / size_u64
    }

    /// Returns the height of this node's tower.
    #[inline]
    fn height(&self) -> usize {
        (self.refs_and_height.load(Relaxed) & HEIGHT_MASK) + 1
    }

    /// Returns the tower as a slice.
    #[inline]
    fn tower(&self) -> &[Atomic<Self>] {
        unsafe { slice::from_raw_parts(&self.pointers[0], self.height()) }
    }

    /// Increments the reference counter of a node and returns `true` on success.
    ///
    /// The reference counter can be incremented only if it is positive.
    ///
    /// If the passed `ptr` is null, this function simply returns.
    #[inline]
    unsafe fn increment(ptr: *const Self) -> bool {
        if ptr.is_null() {
            return true;
        }

        let mut old = (*ptr).refs_and_height.load(Relaxed);
        loop {
            if old & !HEIGHT_MASK == 0 {
                return false;
            }

            let new = old.checked_add(1 << HEIGHT_BITS)
                .expect("reference counter overflow");

            match (*ptr)
                .refs_and_height
                .compare_exchange_weak(old, new, AcqRel, Acquire)
            {
                Ok(_) => return true,
                Err(o) => old = o,
            }
        }
    }
}

impl<K: Send + 'static, V> Node<K, V> {
    /// Decrements the reference counter of a node, scheduling it for GC if the count becomes zero.
    ///
    /// If the passed `ptr` is null, this function simply returns.
    #[inline]
    unsafe fn decrement(ptr: *const Self) {
        if let Some(node) = ptr.as_ref() {
            if node.refs_and_height.fetch_sub(1 << HEIGHT_BITS, AcqRel) >> HEIGHT_BITS == 1 {
                Self::finalize(ptr);
            }
        }
    }

    /// Drops the value of a node and defers destruction of the key and deallocation.
    #[cold]
    unsafe fn finalize(ptr: *const Self) {
        let ptr = ptr as *mut Self;

        // The value can only be read if the reference counter is positive.
        ptr::drop_in_place(&mut (*ptr).value);

        epoch::pin().defer(move || {
            // The key can be read even if the reference counter is zero, assuming that the current
            // thread is pinned. In order to safely drop the key, we have to first wait until all
            // currently pinned threads get unpinned.
            ptr::drop_in_place(&mut (*ptr).key);

            // Finally, deallocate the memory occupied by the node.
            Node::dealloc(ptr)
        });
    }
}

pub struct Skiplist<K, V> {
    head: *mut Node<K, V>, // !Send + !Sync
    seed: AtomicUsize,
}

unsafe impl<K: Send + Sync, V: Send + Sync> Send for Skiplist<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for Skiplist<K, V> {}

// TODO: impl drop for skiplist

impl<K, V> Skiplist<K, V> {
    /// Returns a new, empty skiplist.
    pub fn new() -> Skiplist<K, V> {
        Skiplist {
            head: unsafe { Node::alloc(MAX_HEIGHT) },
            seed: AtomicUsize::new(1),
        }
    }

    /// Returns `true` if the skiplist is empty.
    pub fn is_empty(&self) -> bool {
        unsafe {
            (*self.head).tower()[0]
                .load(SeqCst, epoch::unprotected())
                .is_null()
        }
    }

    /// Generates a random height and returns it.
    fn random_height(&self) -> usize {
        // From "Xorshift RNGs" by George Marsaglia.
        let mut num = self.seed.load(Relaxed);
        num ^= num << 13;
        num ^= num >> 17;
        num ^= num << 5;
        self.seed.store(num, Relaxed);

        let height = cmp::min(num.trailing_zeros() as usize + 1, MAX_HEIGHT) * 2 / 2;
        debug_assert!(1 <= height && height <= MAX_HEIGHT);
        height
    }
}

/// A search result.
///
/// The result indicates whether the key was found, as well as what were the adjacent nodes to the
/// key on each level of the skiplist.
struct Search<'g, K: 'g, V: 'g> {
    /// This flag is `true` if a node with the search key was found.
    ///
    /// More precisely, it will be `true` when `right[0].deref().key` equals the search key.
    found: bool,

    /// Adjacent nodes with smaller keys.
    left: [&'g Node<K, V>; MAX_HEIGHT],

    /// Adjacent nodes with equal or greater keys.
    right: [Shared<'g, Node<K, V>>; MAX_HEIGHT],
}

pub struct InsertError<'a, K: Send + 'static, V: 'a> {
    pub key: K,
    pub value: V,
    pub cursor: Cursor<'a, K, V>,
}

impl<K: Ord + Send + 'static, V> Skiplist<K, V> {
    pub fn cursor(&self) -> Cursor<K, V> {
        Cursor::new(self, ptr::null())
    }

    pub fn insert(&self, key: K, value: V) -> Result<Cursor<K, V>, InsertError<K, V>> {
        // TODO: what about panic safety?
        let guard = &epoch::pin();

        // First try searching for the key.
        let mut s = self.search(Some(&key), guard);

        // If a node with the key was found, let's try creating a cursor pointing at it.
        if s.found {
            let r = s.right[0];

            unsafe {
                // Try incrementing its reference count.
                if Node::increment(r.as_raw()) {
                    // Success!
                    return Err(InsertError {
                        key,
                        value,
                        cursor: Cursor::new(self, r.deref()),
                    });
                }

                // If we couldn't increment the reference count, that means that someone has just
                // deleted the node.
            }
        }

        // Create a new node.
        let height = self.random_height();
        let (node, n) = unsafe {
            let n = Node::<K, V>::alloc(height);

            // Write the key and the value into the node.
            ptr::write(&mut (*n).key, key);
            ptr::write(&mut (*n).value, value);

            // Increment the reference count twice to account for:
            // 1. The cursor that will be returned.
            // 2. The link at the level 0 of the tower.
            (*n).refs_and_height.fetch_add(2 << HEIGHT_BITS, AcqRel);

            (Shared::<Node<K, V>>::from(n as *const _), &*n)
        };

        loop {
            // Set the lowest successor of `n` to `s.right[0]`.
            n.tower()[0].store(s.right[0], SeqCst);

            // Try installing the new node into the skiplist.
            if s.left[0].tower()[0]
                .compare_and_set(s.right[0], node, SeqCst, guard)
                .is_ok()
            {
                break;
            }

            // We failed. Let's search for the key and try again.
            s = self.search(Some(&n.key), guard);

            // Have we found a node with the key this time?
            if s.found {
                let r = s.right[0];

                unsafe {
                    // Try incrementing its reference count.
                    if Node::increment(r.as_raw()) {
                        // Success! Let's deallocate the new node and return an `InsertError`.
                        let key = ptr::read(&n.key);
                        let value = ptr::read(&n.value);
                        // TODO: should this be in a panic-safe guard?
                        Node::dealloc(node.as_raw() as *mut Node<K, V>);

                        return Err(InsertError {
                            key,
                            value,
                            cursor: Cursor::new(self, r.deref()),
                        });
                    }

                    // If we couldn't increment the reference count, that means that someone has
                    // just deleted the node.
                }
            }
        }

        // The node was successfully inserted. Let's create a cursor pointing at it.
        let cursor = Cursor::new(self, n as *const _ as *mut _);

        // Build the rest of the tower above level 0.
        'build: for level in 1..height {
            loop {
                // Obtain the predecessor and successor at the current level.
                let pred = s.left[level];
                let succ = s.right[level];

                // Load the current value of the pointer in the tower.
                let next = n.tower()[level].load(SeqCst, guard);

                // If the current pointer is marked, that means another thread is already deleting
                // the node we've just inserted. In that case, let's just stop building the tower.
                if next.tag() == 1 {
                    break 'build;
                }

                // When searching for `key` and traversing the skiplist from the highest level to
                // the lowest, it is possible to observe a node with an equal key at higher levels
                // and then find it missing at the lower levels if it gets removed during traversal.
                // Even worse, it is possible to observe completely different nodes with the exact
                // same key at different levels.
                //
                // Linking the new node to a dead successor with an equal key would create subtle
                // corner cases that would require special care. It's much easier to simply prevent
                // linking two nodes with equal keys.
                //
                // If the successor has the same key as the new node, that means it is marked as
                // deleted and should be unlinked from the skiplist. In that case, let's repeat the
                // search to make sure it gets unlinked and try again.
                if unsafe { succ.as_ref().map(|s| &s.key) } == Some(&n.key) {
                    s = self.search(Some(&n.key), guard);
                    continue;
                }

                // Change the pointer at the current level from `next` to `succ`. If this CAS
                // operation fails, that means another thread has marked the pointer and we should
                // stop building the tower.
                if n.tower()[level]
                    .compare_and_set(next, succ, SeqCst, guard)
                    .is_err()
                {
                    break 'build;
                }

                // Increment the reference count. The current value will always be at least 1
                // because we are holding
                (*n).refs_and_height.fetch_add(1 << HEIGHT_BITS, Relaxed);

                // Try installing the new node at the current level.
                if pred.tower()[level]
                    .compare_and_set(succ, node, SeqCst, guard)
                    .is_ok()
                {
                    // Success! Continue on the next level.
                    break;
                }

                // Installation failed. Decrement the reference count.
                (*n).refs_and_height.fetch_sub(1 << HEIGHT_BITS, Relaxed);

                // We don't have the most up-to-date search results. Repeat the search.
                s = self.search(Some(&n.key), guard);
            }
        }

        // If a pointer in the tower is marked, that means our node is in the process of deletion or
        // already deleted. It is possible that another thread (either partially or completely)
        // deleted the new node while we were building the tower, and just after that we installed
        // the new node at one of the higher levels. In order to undo that installation, we must
        // repeat the search, which will unlink the new node at that level.
        if n.tower()[height - 1].load(SeqCst, guard).tag() == 1 {
            self.search(Some(&n.key), guard);
        }

        Ok(cursor)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<Cursor<K, V>>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        unimplemented!()
    }

    // TODO: Return Option<Cursor<K, V>>
    pub fn remove<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let guard = &epoch::pin();

        // Try searching for the key.
        let s = self.search(Some(key), guard);
        if !s.found {
            return false;
        }

        let node = s.right[0];
        let n = unsafe { node.deref() };
        let height = n.height();

        for level in (0..height).rev() {
            let next = n.tower()[level].fetch_or(1, SeqCst, guard);

            if level == 0 && next.tag() == 1 {
                return false;
            }
        }

        for level in (0..height).rev() {
            let succ = n.tower()[level].load(SeqCst, guard).with_tag(0);

            if s.left[level].tower()[level]
                .compare_and_set(node, succ, SeqCst, guard)
                .is_ok()
            {
                unsafe {
                    Node::decrement(n);
                }
            } else {
                self.search(Some(key), guard);
                break;
            }
        }

        true
    }

    fn search<'g, Q>(&self, key: Option<&Q>, guard: &'g Guard) -> Search<'g, K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut s = Search {
            found: false,
            left: unsafe { mem::zeroed() },
            right: unsafe { mem::zeroed() },
        };

        'search: loop {
            // The current node we're at.
            let mut node = unsafe { &*self.head };

            // Traverse the skiplist from the highest to the lowest level.
            for level in (0..MAX_HEIGHT).rev() {
                let mut pred = node;
                let mut curr = pred.tower()[level].load(SeqCst, guard);

                // If `curr` is marked, that means `pred` is deleted and we have to restart the
                // search.
                if curr.tag() == 1 {
                    continue 'search;
                }

                // Iterate through the current level until we reach a node with a key greater than
                // or equal to `key`.
                while let Some(c) = unsafe { curr.as_ref() } {
                    let succ = c.tower()[level].load(SeqCst, guard);

                    if succ.tag() == 1 {
                        // If `succ` is marked, that means `curr` is deleted. Let's try unlinking
                        // it from the skiplist at this level.
                        match pred.tower()[level].compare_and_set(
                            curr,
                            succ.with_tag(0),
                            SeqCst,
                            guard,
                        ) {
                            Ok(_) => {
                                // On success, decrement the reference counter of `curr` and
                                // continue searching through the current level.
                                unsafe {
                                    Node::decrement(curr.as_raw());
                                }
                                curr = succ.with_tag(0);
                                continue;
                            }
                            Err(_) => {
                                // On failure, we cannot do anything reasonable to continue
                                // searching from the current position. Restart the search.
                                continue 'search;
                            }
                        }
                    }

                    // If `curr` contains a key that is greater than or equal to `key`, we're done
                    // with this level.
                    if key.map(|k| c.key.borrow() >= k) == Some(true) {
                        break;
                    }

                    // Move one step forward.
                    pred = c;
                    curr = succ;
                }

                // Store the position at the current level into the result.
                s.left[level] = pred;
                s.right[level] = curr;

                node = pred;
            }

            // Check if we have found a node with key equal to `key`.
            s.found = unsafe {
                s.right[0].as_ref().map(|r| Some(r.key.borrow()) == key) == Some(true)
            };

            return s;
        }
    }
}

pub struct Cursor<'a, K: Send + 'static, V: 'a> {
    parent: &'a Skiplist<K, V>,
    node: *const Node<K, V>,
}

unsafe impl<'a, K: Send + Sync, V: Send + Sync> Send for Cursor<'a, K, V> {}
unsafe impl<'a, K: Send + Sync, V: Send + Sync> Sync for Cursor<'a, K, V> {}

impl<'a, K: Ord + Send + 'static, V> Cursor<'a, K, V> {
    fn new(parent: &'a Skiplist<K, V>, node: *const Node<K, V>) -> Self {
        Cursor {
            parent: parent,
            node: node,
        }
    }

    /// Returns `true` if the cursor is null.
    pub fn is_null(&self) -> bool {
        self.node.is_null()
    }

    /// Returns `true` if the cursor is pointing at an alive element.
    pub fn is_valid(&self) -> bool {
        unsafe {
            self.node
                .as_ref()
                .map(|r| r.tower()[0].load(SeqCst, epoch::unprotected()).tag() == 0)
                .unwrap_or(false)
        }
    }

    /// Returns the key of the element pointed to by the cursor.
    pub fn key(&self) -> Option<&K> {
        unsafe { self.node.as_ref().map(|r| &r.key) }
    }

    /// Returns the value of the element pointed to by the cursor.
    pub fn value(&self) -> Option<&V> {
        unsafe { self.node.as_ref().map(|r| &r.value) }
    }

    /// Returns a reference to the skiplist owning this cursor.
    pub fn parent(&self) -> &Skiplist<K, V> {
        self.parent
    }

    // TODO: what happens if this one is removed and there is a new element with the same key?
    /// Moves the cursor to the next element in the skiplist.
    pub fn next(&mut self) {
        let node_ref = unsafe { self.node.as_ref() };

        loop {
            let guard = &epoch::pin();

            let ptr = match node_ref {
                None => {
                    let head = unsafe { &*self.parent.head };
                    head.tower()[0].load(SeqCst, guard)
                }
                Some(node) => {
                    let succ = node.tower()[0].load(SeqCst, guard);
                    if succ.tag() == 0 {
                        succ
                    } else {
                        let s = self.parent.search(Some(&node.key), guard);
                        if s.found {
                            unsafe { s.right[0].deref().tower()[0].load(SeqCst, guard) }
                        } else {
                            s.right[0]
                        }
                    }
                }
            };

            if ptr.tag() == 0 {
                let success = unsafe {
                    match ptr.as_ref() {
                        None => true,
                        Some(c) => Node::increment(c),
                    }
                };

                if success {
                    if let Some(node) = node_ref {
                        unsafe {
                            Node::decrement(node);
                        }
                    }
                    self.node = ptr.as_raw();
                    return;
                }
            }
        }
    }

    // TODO: what happens if this one is removed and there is a new element with the same key?
    /// Moves the cursor to the previous element in the skiplist.
    pub fn prev(&mut self) {
        loop {
            let guard = &epoch::pin();
            let s = self.parent.search(self.key(), guard);
            let pred = s.left[0];

            unsafe {
                if ptr::eq(pred, self.parent.head) {
                    Node::decrement(self.node);
                    self.node = ptr::null();
                    break;
                }

                if Node::increment(pred) {
                    Node::decrement(self.node);
                    self.node = pred;
                    break;
                }
            }
        }
    }

    /// Positions the cursor to the first element with key equal to or greater than `key`.
    pub fn seek<Q>(&mut self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        loop {
            let guard = &epoch::pin();
            let s = self.parent.search(Some(key.borrow()), guard);
            let node = s.right[0].as_raw();

            unsafe {
                if Node::increment(node) {
                    Node::decrement(self.node);
                    self.node = node;
                    return s.found;
                }
            }
        }
    }

    pub fn remove(&self) -> bool {
        unimplemented!()
    }
}

impl<'a, K: Send + 'static, V> Drop for Cursor<'a, K, V> {
    fn drop(&mut self) {
        if !self.node.is_null() {
            unsafe { Node::decrement(self.node) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::sync::Arc;
    use std::sync::atomic::Ordering::{Relaxed, SeqCst};
    use std::ptr;

    #[test]
    fn foo() {
        let s = Skiplist::new();
        let my = Arc::new(s);

        use std::time::{Duration, Instant};
        let now = Instant::now();

        const T: usize = 1;
        let mut v = (0..T)
            .map(|mut t| {
                let my = my.clone();
                thread::spawn(move || {
                    let mut num = t as u32;
                    for i in 0..1000_000 / T {
                        num = num.wrapping_mul(17).wrapping_add(255);
                        my.insert(num, !num);
                    }
                })
            })
            .collect::<Vec<_>>();
        for h in v.drain(..) {
            h.join().unwrap();
        }
        // v.extend((0..T).map(|mut t| {
        //     let my = my.clone();
        //     thread::spawn(move || {
        //         let mut num = t as u32;
        //         for i in 0..1_000_000 / T {
        //             num = num.wrapping_mul(17).wrapping_add(255);
        //             my.remove(&num);
        //         }
        //     })
        // }));
        // for h in v {
        //     h.join().unwrap();
        // }

        let elapsed = now.elapsed();

        // let elapsed = now.elapsed();
        println!(
            "seconds: {:.3}",
            elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1e9
        );
        // println!("LEN: {}", my.count());

        let now = Instant::now();
        let mut c = my.cursor();
        c.next();
        let mut cnt = 0;
        while !c.is_null() {
            cnt += 1;
            c.next();
        }
        println!("cnt = {}", cnt);
        let elapsed = now.elapsed();
        println!(
            "iteration seconds: {:.3}",
            elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1e9
        );

        my.insert(10, 0);
        my.insert(20, 0);
        my.insert(30, 0);

        let mut c = my.cursor();
        println!("{:?}", c.seek(&0));
        println!("-> {:?}", c.key());
    }

    // #[test]
    // fn it_works() {
    //     let s = Skiplist::new();
    //     let my = Arc::new(s);
    //
    //     use std::time::{Duration, Instant};
    //     let now = Instant::now();
    //
    //     const T: usize = 1;
    //     let mut v = (0..T)
    //         .map(|mut t| {
    //             let my = my.clone();
    //             thread::spawn(move || {
    //                 let mut num = t as u32;
    //                 for i in 0..1_000_000 / T {
    //                     num = num.wrapping_mul(17).wrapping_add(255);
    //                     my.insert(num, !num);
    //                 }
    //             })
    //         })
    //         .collect::<Vec<_>>();
    //     // v.extend((0..T).map(|mut t| {
    //     //     let my = my.clone();
    //     //     thread::spawn(move || {
    //     //         let mut num = t as u32;
    //     //         for i in 0 .. 1_000_000 / T {
    //     //             num = num.wrapping_mul(17).wrapping_add(255);
    //     //             my.remove(&num);
    //     //         }
    //     //     })
    //     // }));
    //     for h in v {
    //         h.join();
    //     }
    //     // let mut num = 0 as u32;
    //     // for i in 0 .. 1_000_000 / T {
    //     //     num = num.wrapping_mul(17).wrapping_add(255);
    //     //     my.remove(&num);
    //     // }
    //
    //     let elapsed = now.elapsed();
    //     let now = Instant::now();
    //
    //     let mut x = my.cursor();
    //     x.front();
    //     let mut steps = 0;
    //     while !x.is_null() {
    //         // unsafe {
    //         //     let node = x.node.as_ref().unwrap();
    //         //     assert_eq!(node.data.refs.load(Relaxed), node.data.height as usize + 1);
    //         // }
    //         x.next();
    //         steps += 1;
    //     }
    //     println!("STEPS: {}", steps);
    //
    //     // let elapsed = now.elapsed();
    //     println!(
    //         "seconds: {:.3}",
    //         elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1e9
    //     );
    //     println!("LEN: {}", my.count());
    // }
}
