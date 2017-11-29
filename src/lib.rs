extern crate crossbeam_epoch as epoch;

use std::borrow::Borrow;
use std::cmp;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release, SeqCst};

use epoch::{Atomic, Guard, Owned, Shared};

// TODO: a test where comparison function sometimes panics
// TODO: Explain why TrustedOrd is not required
// TODO: In remove directly relink instead of calling search_level?

// TODO: Make sure not to execute user defined code within an epoch!! not even Drop for K and V

// TODO: Idea: if using hazard pointers, we don't need the height - we can just tag the last pointer

// TODO: See ThinArc in Servo

const HEIGHT: usize = 1 << HEIGHT_BITS;
const HEIGHT_BITS: usize = 5;
const HEIGHT_MASK: usize = (1 << HEIGHT_BITS) - 1;

/// TODO: Explain: key must be close to tower, and tower must be the last
#[repr(C)]
struct Node<K, V> {
    value: V,
    key: K,
    refs_and_height: AtomicUsize,
    pointers: [Atomic<Node<K, V>>; 1],
}

unsafe impl<K, V> Send for Node<K, V> {}
unsafe impl<K, V> Sync for Node<K, V> {}

impl<K, V> Node<K, V> {
    fn size_with_height(height: usize) -> usize {
        assert!(1 <= height && height <= HEIGHT);

        let base_size = mem::size_of::<Self>();
        let ptr_size = mem::size_of::<Atomic<Self>>();
        base_size
            .checked_add(ptr_size.checked_mul(height - 1).unwrap())
            .unwrap()
    }

    // TODO: why unsafe?
    unsafe fn alloc(height: usize) -> *mut Self {
        let size_u64 = mem::size_of::<u64>();
        let align_u64 = mem::align_of::<u64>();

        let size = Self::size_with_height(height);
        let align = mem::align_of::<Self>();

        let ptr = if align <= align_u64 {
            let cap = (size.checked_add(size_u64).unwrap() - 1) / size_u64;
            let mut v = Vec::<u64>::with_capacity(cap);
            let ptr = v.as_mut_ptr();
            mem::forget(v);
            ptr as *mut Self
        } else {
            let cap = (size.checked_add(size_u64).unwrap() - 1) / size;
            let mut v = Vec::<Self>::with_capacity(cap);
            let ptr = v.as_mut_ptr();
            mem::forget(v);
            ptr
        };

        (*ptr).refs_and_height = AtomicUsize::new((height << HEIGHT_BITS) | height);
        ptr::write_bytes(&mut (*ptr).pointers[0], 0, height);

        ptr
    }

    unsafe fn destroy(ptr: *mut Self) {
        let size_u64 = mem::size_of::<u64>();
        let align_u64 = mem::align_of::<u64>();

        let size = Self::size_with_height((*ptr).height());
        let align = mem::align_of::<Self>();

        ptr::drop_in_place(&mut (*ptr).key);
        ptr::drop_in_place(&mut (*ptr).value);

        if align <= align_u64 {
            let cap = (size.checked_add(size_u64).unwrap() - 1) / size_u64;
            drop(Vec::from_raw_parts(ptr as *mut u64, 0, cap));
        } else {
            let cap = (size.checked_add(size_u64).unwrap() - 1) / size;
            drop(Vec::from_raw_parts(ptr, 0, cap));
        }
    }

    fn height(&self) -> usize {
        self.refs_and_height.load(Relaxed) & HEIGHT_MASK
    }

    unsafe fn tower(&self, level: usize) -> &Atomic<Self> {
        // TODO: debug assert bound checking
        &*self.pointers.as_ptr().offset(level as isize)
    }

    unsafe fn acquire(ptr: Shared<Self>) -> bool {
        let mut old = ptr.deref().refs_and_height.load(Relaxed);

        while old & !HEIGHT_MASK > 0 {
            // TODO: explain that we need checked_add for safety (with mem::forget it's possible to
            // overflow)
            let new = old.checked_add(1 << HEIGHT_BITS).unwrap();

            match ptr.deref().refs_and_height
                .compare_exchange_weak(old, new, AcqRel, Acquire)
            {
                Ok(_) => return true,
                Err(o) => old = o,
            }
        }
        false
    }

    unsafe fn release(ptr: Shared<Self>) {
        if ptr.deref().refs_and_height.fetch_sub(1 << HEIGHT_BITS, AcqRel) >> HEIGHT_BITS == 1 {
            let guard = epoch::pin();

            unsafe {
                guard.defer(move || Node::destroy(ptr.as_raw() as *mut Self));
            }
        }
    }
}

pub struct Skiplist<K, V> {
    head: *mut Node<K, V>, // !Send + !Sync
    seed: AtomicUsize,
}

unsafe impl<K: Send + Sync, V: Send + Sync> Send for Skiplist<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for Skiplist<K, V> {}

// TODO: impl drop for skiplist

impl<K: Ord, V> Skiplist<K, V> {
    /// Returns a new, empty skiplist.
    pub fn new() -> Self {
        Skiplist {
            head: unsafe { Node::alloc(HEIGHT) },
            seed: AtomicUsize::new(1),
        }
    }

    // TODO: new()
    // TODO: count(&self)
    // TODO: cursor(&self) -> Cursor<K, V>
    // TODO: insert(&self, k, v) -> Result<Cursor<K, V>, Cursor<K, V>>
    // TODO: remove(&self, k, v) -> Option<Cursor<K, V>>
    // TODO: front(&self) -> Cursor<K, V>
    // TODO: back(&self) -> Cursor<K, V>
    // TODO: get(&self, k) -> Option<Cursor<K, V>>

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

    unsafe fn search_level<'g, Q>(
        &self,
        level: usize,
        key: &Q,
        from: &'g Node<K, V>,
        guard: &'g Guard,
    ) -> Result<(&'g Node<K, V>, Shared<'g, Node<K, V>>), ()>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut pred = from;
        let mut curr = pred.tower(level).load(SeqCst, guard);
        if curr.tag() == 1 {
            return Err(());
        }

        while let Some(c) = curr.as_ref() {
            let succ = c.tower(level).load(SeqCst, guard);

            if succ.tag() == 1 {
                match pred.tower(level).compare_and_set(curr, succ.with_tag(0), SeqCst, guard) {
                    Ok(_) => Node::release(curr),
                    Err(_) => return Err(()),
                }
                curr = succ.with_tag(0);
            } else {
                if c.key.borrow() >= key {
                    break;
                }
                pred = c;
                curr = succ;
            }
        }
        Ok((pred, curr))
    }

    unsafe fn search<'g, Q>(
        &self,
        key: &Q,
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
        let (mut left, mut right) =
            mem::uninitialized::<([&Node<K, V>; HEIGHT], [Shared<Node<K, V>>; HEIGHT])>();
        'search: loop {
            let mut curr = &*self.head;

            for level in (0..HEIGHT).rev() {
                match self.search_level(level, key, curr, guard) {
                    Ok((l, r)) => {
                        left[level] = l;
                        right[level] = r;
                    }
                    Err(()) => continue 'search,
                }
                curr = left[level];
            }

            let found = match right[0].as_ref() {
                None => false,
                Some(r) => r.key.borrow() == key,
            };
            return (found, left, right);
        }
    }

    pub fn insert(&self, key: K, value: V) -> Result<Cursor<K, V>, Cursor<K, V>> {
        // TODO: what about panic safety?
        let guard = &epoch::pin();

        // self.garbage.collect(guard);

        unsafe {
            let (found, mut left, mut right) = self.search(&key, guard);
            if found {
                let r = right[0];
                if Node::acquire(r) {
                    return Err(Cursor::new(self, r.deref()));
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

                let (found, l, r) = self.search(&c.key, guard);
                left = l;
                right = r;
                if found {
                    let r = right[0];
                    if Node::acquire(r) {
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

                    if let Some(s) = succ.as_ref() {
                        if &s.key == &c.key {
                            let (_, l, r) = self.search(&c.key, guard);
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
                        let (_, l, r) = self.search(&c.key, guard);
                        left = l;
                        right = r;
                    }
                }
            }

            for _ in built..height {
                Node::release(curr);
            }

            if c.tower(0).load(SeqCst, guard).tag() == 1 {
                self.search(&c.key, guard);
            }

            Ok(Cursor::new(self, c as *const _ as *mut _))
        }
    }

    pub fn remove<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        let guard = &epoch::pin();

        unsafe {
            let (found, mut left, mut right) = self.search(key, guard);
            if !found {
                return false;
            }

            let curr = right[0].deref();
            let height = curr.height();

            for level in (0..height).rev() {
                let mut next = curr.tower(level).load(SeqCst, guard);
                while next.tag() == 0 {
                    match curr.tower(level).compare_and_set(next, next.with_tag(1), SeqCst, guard) {
                        Ok(_) => break,
                        Err(err) => next = err.current,
                    }
                }

                if level == 0 && next.tag() == 0 {
                    // TODO: try relinking manually, decrement by the number of successful relinks
                    for i in (0..height).rev() {
                        if self.search_level(i, key, left[i], guard).is_err() {
                            self.search(key, guard);
                            break;
                        }
                    }
                    return true;
                }
            }
        }

        false
    }

    pub fn cursor(&self) -> Cursor<K, V> {
        Cursor::new(self, ptr::null())
    }
}

pub struct Cursor<'a, K: 'a, V: 'a> {
    parent: &'a Skiplist<K, V>,
    node: *const Node<K, V>,
}

unsafe impl<'a, K: Send + Sync, V: Send + Sync> Send for Cursor<'a, K, V> {}
unsafe impl<'a, K: Send + Sync, V: Send + Sync> Sync for Cursor<'a, K, V> {}

impl<'a, K: Ord, V> Cursor<'a, K, V> {
    fn new(parent: &'a Skiplist<K, V>, node: *const Node<K, V>) -> Self {
        Cursor {
            parent: parent,
            node: node,
        }
    }

    // TODO: parent(&self) -> &SkiplistMap<K, V>
    // TODO: seek(&mut self, k) -> bool
    // TODO: first(&mut self)
    // TODO: last(&mut self)
    // TODO: prev(&mut self)
    // TODO: next(&mut self)
    // TODO: remove(&self) -> bool
    // TODO: is_valid(&self) -> bool

    // pub fn front(&mut self) {
    //     let node_ref = unsafe { self.node.as_ref() };
    //
    //     loop {
    //         let done = epoch::pin(|guard| {
    //
    //             let head = unsafe { &*self.parent.head };
    //             let candidate = head.tower(0).load(guard);
    //
    //             if candidate.tag() == 0 {
    //                 let success = match candidate.as_ref() {
    //                     None => true,
    //                     Some(c) => c.inc(),
    //                 };
    //
    //                 if success {
    //                     node_ref.map(|node| node.dec(&self.parent.garbage));
    //                     self.node = candidate.as_raw();
    //                     return true;
    //                 }
    //             }
    //             false
    //         });
    //
    //         if done {
    //             break;
    //         }
    //     }
    // }

    pub fn back(&mut self) {
        unimplemented!()
    }

    // pub fn next(&mut self) {
    //     let node_ref = unsafe { self.node.as_ref() };
    //
    //     loop {
    //         let done = epoch::pin(|guard| {
    //
    //             let candidate = match node_ref {
    //                 None => {
    //                     let head = unsafe { &*self.parent.head };
    //                     head.tower(0).load(guard)
    //                 }
    //                 Some(node) => {
    //                     let succ = node.tower(0).load(guard);
    //                     if succ.tag() == 1 {
    //                         let (found, _left, right) = self.parent.search(&node.key, guard);
    //                         if found {
    //                             right[0].unwrap().tower(0).load(guard)
    //                         } else {
    //                             right[0]
    //                         }
    //                     } else {
    //                         succ
    //                     }
    //                 }
    //             };
    //
    //             if candidate.tag() == 0 {
    //                 let success = match candidate.as_ref() {
    //                     None => true,
    //                     Some(c) => c.inc(),
    //                 };
    //
    //                 if success {
    //                     node_ref.map(|node| node.dec(&self.parent.garbage));
    //                     self.node = candidate.as_raw();
    //                     return true;
    //                 }
    //             }
    //             false
    //         });
    //
    //         if done {
    //             break;
    //         }
    //     }
    // }

    // pub fn previous(&mut self) {
    //     let node_ref = unsafe { self.node.as_ref() };
    //
    //     loop {
    //         let done = epoch::pin(|guard| {
    //
    //             let candidate = match node_ref {
    //                 None => {
    //                     unimplemented!()
    //                 }
    //                 Some(node) => {
    //                     let succ = node.tower(0).load(guard);
    //                     if succ.tag() == 1 {
    //                         let (found, _left, right) = self.parent.search(&node.key, guard);
    //                         if found {
    //                             right[0].unwrap().tower(0).load(guard)
    //                         } else {
    //                             right[0]
    //                         }
    //                     } else {
    //                         succ
    //                     }
    //                 }
    //             };
    //
    //             if candidate.tag() == 0 {
    //                 let success = match candidate.as_ref() {
    //                     None => true,
    //                     Some(c) => c.inc(),
    //                 };
    //
    //                 if success {
    //                     node_ref.map(|node| node.dec(&self.parent.garbage));
    //                     self.node = candidate.as_raw();
    //                     return true;
    //                 }
    //             }
    //             false
    //         });
    //
    //         if done {
    //             break;
    //         }
    //     }
    // }

    // pub fn is_removed(&self) -> bool {
    //     match unsafe { self.node.as_ref() } {
    //         None => false,
    //         Some(r) => {
    //             let (_ptr, tag) = r.tower(0).load_raw(Acquire);
    //             tag == 1
    //         }
    //     }
    // }

    pub fn is_null(&self) -> bool {
        self.node.is_null()
    }

    pub fn key(&self) -> Option<&K> {
        unsafe { self.node.as_ref().map(|r| &r.key) }
    }

    pub fn value(&self) -> Option<&V> {
        unsafe { self.node.as_ref().map(|r| &r.value) }
    }

    pub fn seek<Q>(&mut self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        unimplemented!()
    }

    pub fn remove(&self) -> bool {
        unimplemented!()
    }
}

impl<'a, K, V> Drop for Cursor<'a, K, V> {
    fn drop(&mut self) {
        // unsafe { self.node.as_ref().map(|node| node.dec(&self.parent.garbage)); }
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
                    for i in 0..1_000_000 / T {
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
        //         for i in 0 .. 1_000_000 / T {
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
        println!("LEN: {}", my.count());
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
