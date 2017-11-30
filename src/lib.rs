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

const HEIGHT: usize = 1 << HEIGHT_BITS;
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
    /// The returned node will start with reference count equal to `height` and the tower will be
    /// initialized with null pointers.
    fn alloc(height: usize) -> *mut Self {
        let cap = Self::size_in_u64s(height);
        let mut v = Vec::<u64>::with_capacity(cap);
        let ptr = v.as_mut_ptr() as *mut Self;
        mem::forget(v);

        unsafe {
            (*ptr).refs_and_height = AtomicUsize::new((height << HEIGHT_BITS) | (height - 1));
            ptr::write_bytes(&mut (*ptr).pointers[0], 0, height);
            ptr
        }
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
        assert!(1 <= height && height <= HEIGHT);
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

    /// Returns a reference to the atomic pointer at the specified `level`.
    ///
    /// Argument `level` must be in the range `0..self.height()`.
    #[inline]
    fn tower(&self, level: usize) -> &Atomic<Self> {
        let tower = unsafe { slice::from_raw_parts(&self.pointers[0], self.height()) };
        &tower[level]
    }

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

            let new = old.checked_add(1 << HEIGHT_BITS).expect("reference count overflow");

            match (*ptr).refs_and_height
                .compare_exchange_weak(old, new, AcqRel, Acquire)
            {
                Ok(_) => return true,
                Err(o) => old = o,
            }
        }
    }
}

impl<K: Send + 'static, V> Node<K, V> {
    #[inline]
    unsafe fn decrement(ptr: *const Self) {
        if let Some(node) = ptr.as_ref() {
            if node.refs_and_height.fetch_sub(1 << HEIGHT_BITS, AcqRel) >> HEIGHT_BITS == 1 {
                Self::finalize(ptr);
            }
        }
    }

    #[cold]
    unsafe fn finalize(ptr: *const Self) {
        let ptr = ptr as *mut Self;
        ptr::drop_in_place(&mut (*ptr).value);

        epoch::pin().defer(move || {
            ptr::drop_in_place(&mut (*ptr).key);
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
    pub fn new() -> Self {
        Skiplist {
            head: unsafe { Node::alloc(HEIGHT) },
            seed: AtomicUsize::new(1),
        }
    }

    // TODO: new()
    // TODO: count(&self), or len(&self) with scalabale counter
    // TODO: lower_bound / upper_bound, maybe seek? find first with key >= target
    // TODO: get(&self, k) -> Option<Cursor<K, V>>
    // TODO: next/prev/
    // TODO: is_valid()

    /// Counts the elements in the skiplist.
    ///
    /// This method traverses the whole skiplist, which takes linear time.
    pub fn count(&self) -> usize {
        // TODO: tagged nodes, relinking, etc.
        // TODO: shorter pinning
        let guard = &epoch::pin();

        unsafe {
            let mut curr = unsafe { (*self.head).tower(0).load(SeqCst, guard) };

            let mut count = 0;
            while let Some(c) = curr.as_ref() {
                curr = c.tower(0).load(SeqCst, guard);
                count += 1;
            }
            count
        }
    }

    fn random_height(&self) -> usize {
        // TODO: From "Xorshift RNGs" by George Marsaglia.
        let mut num = self.seed.load(Relaxed);
        num ^= num << 13;
        num ^= num >> 17;
        num ^= num << 5;
        self.seed.store(num, Relaxed);
        cmp::min(num.trailing_zeros() as usize + 1, HEIGHT)
    }
}

impl<K: Ord + Send + 'static, V> Skiplist<K, V> {
    // TODO: return Search<'g, K, V>
    fn search<'g, Q>(
        &self,
        key: Option<&Q>,
        guard: &'g Guard,
    ) -> (
        bool,
        [&'g Node<K, V>; HEIGHT],
        [Shared<'g, Node<K, V>>; HEIGHT],
    )
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (mut left, mut right) = unsafe {
            mem::uninitialized::<([&Node<K, V>; HEIGHT], [Shared<Node<K, V>>; HEIGHT])>()
        };

        'search: loop {
            let mut node = unsafe { &*self.head };

            for level in (0..HEIGHT).rev() {
                let mut pred = node;
                let mut curr = pred.tower(level).load(SeqCst, guard);

                if curr.tag() == 1 {
                    continue 'search;
                }

                while let Some(c) = unsafe { curr.as_ref() } {
                    let succ = c.tower(level).load(SeqCst, guard);

                    if succ.tag() == 1 {
                        match pred.tower(level).compare_and_set(curr, succ.with_tag(0), SeqCst, guard) {
                            Ok(_) => unsafe { Node::decrement(curr.as_raw()) },
                            Err(_) => continue 'search,
                        }
                        curr = succ.with_tag(0);
                    } else {
                        if let Some(key) = key {
                            if c.key.borrow() >= key {
                                break;
                            }
                        }
                        pred = c;
                        curr = succ;
                    }
                }

                left[level] = pred;
                right[level] = curr;

                node = pred;
            }

            let found = match unsafe { right[0].as_ref() } {
                None => false,
                Some(r) => Some(r.key.borrow()) == key,
            };
            return (found, left, right);
        }
    }

    // TODO: return Insert<'a, K, V>
    pub fn insert(&self, key: K, value: V) -> Result<Cursor<K, V>, Cursor<K, V>> {
        // TODO: what about panic safety?
        let guard = &epoch::pin();

        let (found, mut left, mut right) = self.search(Some(&key), guard);
        if found {
            let r = right[0];
            unsafe {
                if Node::increment(r.as_raw()) {
                    return Err(Cursor::new(self, r.deref()));
                }
            }
        }

        let height = self.random_height();
        let (curr, c) = unsafe {
            let n = Node::<K, V>::alloc(height);
            (*n).key = key;
            (*n).value = value;
            (*n).refs_and_height.fetch_add(1 << HEIGHT_BITS, AcqRel);
            (Shared::<Node<K, V>>::from(n as *const _), &*n)
        };

        loop {
            c.tower(0).store(right[0], SeqCst);
            if left[0].tower(0).compare_and_set(right[0], curr, SeqCst, guard).is_ok() {
                break;
            }

            let (found, l, r) = self.search(Some(&c.key), guard);
            left = l;
            right = r;
            if found {
                let r = right[0];
                if unsafe { Node::increment(r.as_raw()) } {
                    // TODO: deallocate curr (should probably be in a panic-safe guard)
                    return Err(Cursor::new(self, r.as_raw() as *mut _));
                }
            }
        }

        let mut built = 1;
        'build: for level in 1..height {
            loop {
                let pred = left[level];
                let succ = right[level];

                // TODO: Explain why this if goes before the following if
                let next = c.tower(level).load(SeqCst, guard);
                if next.tag() == 1 {
                    break 'build;
                }

                if let Some(s) = unsafe { succ.as_ref() } {
                    if &s.key == &c.key {
                        let (_, l, r) = self.search(Some(&c.key), guard);
                        left = l;
                        right = r;
                        continue;
                    }
                }

                if next.as_raw() != succ.as_raw() {
                    if c.tower(level).compare_and_set(next, succ, SeqCst, guard).is_err() {
                        break 'build;
                    }
                }

                if pred.tower(level).compare_and_set(succ, curr, SeqCst, guard).is_ok() {
                    built += 1;
                    break;
                } else {
                    let (_, l, r) = self.search(Some(&c.key), guard);
                    left = l;
                    right = r;
                }
            }
        }

        for _ in built..height {
            unsafe { Node::decrement(curr.as_raw()); }
        }

        if c.tower(0).load(SeqCst, guard).tag() == 1 {
            self.search(Some(&c.key), guard);
        }

        Ok(Cursor::new(self, c as *const _ as *mut _))
    }

    // TODO: Return Option<Cursor<K, V>>
    pub fn remove<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let guard = &epoch::pin();

        let (found, mut left, mut right) = self.search(Some(key), guard);
        if !found {
            return false;
        }

        let curr = right[0];
        let node = unsafe { curr.deref() };
        let height = node.height();

        for level in (0..height).rev() {
            let next = node.tower(level).fetch_or(1, SeqCst, guard);

            if level == 0 && next.tag() == 1 {
                return false;
            }
        }

        for level in (0..height).rev() {
            let succ = node.tower(level).load(SeqCst, guard).with_tag(0);

            if left[level].tower(level).compare_and_set(curr, succ, SeqCst, guard).is_ok() {
                unsafe { Node::decrement(node); }
            } else {
                self.search(Some(key), guard);
                break;
            }
        }

        true
    }

    pub fn cursor(&self) -> Cursor<K, V> {
        Cursor::new(self, ptr::null())
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

    pub fn parent(&self) -> &Skiplist<K, V> {
        self.parent
    }

    // TODO: what happens if this one is removed and there is a new element with the same key?
    pub fn next(&mut self) {
        let node_ref = unsafe { self.node.as_ref() };

        loop {
            let guard = &epoch::pin();

            let candidate = match node_ref {
                None => {
                    let head = unsafe { &*self.parent.head };
                    head.tower(0).load(SeqCst, guard)
                }
                Some(node) => {
                    let succ = node.tower(0).load(SeqCst, guard);
                    if succ.tag() == 1 {
                        let (found, _left, right) = self.parent.search(Some(&node.key), guard);
                        if found {
                            unsafe { right[0].deref().tower(0).load(SeqCst, guard) }
                        } else {
                            right[0]
                        }
                    } else {
                        succ
                    }
                }
            };

            if candidate.tag() == 0 {
                let success = unsafe {
                    match candidate.as_ref() {
                        None => true,
                        Some(c) => Node::increment(c),
                    }
                };

                if success {
                    if let Some(node) = node_ref {
                        unsafe { Node::decrement(node); }
                    }
                    self.node = candidate.as_raw();
                    return;
                }
            }
        }
    }

    // TODO: what happens if this one is removed and there is a new element with the same key?
    pub fn prev(&mut self) {
        loop {
            let guard = &epoch::pin();
            let (_found, left, _right) = self.parent.search(self.key(), guard);
            let pred = left[0];

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

    pub fn is_valid(&self) -> bool {
        match unsafe { self.node.as_ref() } {
            None => false,
            Some(r) => unsafe {
                r.tower(0).load(SeqCst, epoch::unprotected()).tag() == 0
            }
        }
    }

    pub fn is_null(&self) -> bool {
        self.node.is_null()
    }

    pub fn key(&self) -> Option<&K> {
        unsafe { self.node.as_ref().map(|r| &r.key) }
    }

    pub fn value(&self) -> Option<&V> {
        unsafe { self.node.as_ref().map(|r| &r.value) }
    }

    // pub fn seek<Q>(&mut self, key: &Q) -> bool
    // where
    //     K: Borrow<Q>,
    //     Q: Ord + ?Sized,
    // {
    //     unimplemented!()
    // }
    //
    // pub fn remove(&self) -> bool {
    //     unimplemented!()
    // }
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
        v.extend((0..T).map(|mut t| {
            let my = my.clone();
            thread::spawn(move || {
                let mut num = t as u32;
                for i in 0 .. 1_000_000 / T {
                    num = num.wrapping_mul(17).wrapping_add(255);
                    my.remove(&num);
                }
            })
        }));
        for h in v {
            h.join().unwrap();
        }

        let elapsed = now.elapsed();

        // let elapsed = now.elapsed();
        println!(
            "seconds: {:.3}",
            elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1e9
        );
        println!("LEN: {}", my.count());

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
