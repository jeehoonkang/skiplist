use std::borrow::Borrow;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, SeqCst};

use epoch::{self, Atomic, Guard, Shared};

const MAX_HEIGHT: usize = 32;
const HEIGHT_BITS: usize = 5;
const HEIGHT_MASK: usize = (1 << HEIGHT_BITS) - 1;

/// A skip list node.
///
/// This struct is marked with `repr(C)` so that the specific order of fields can be enforced.
/// It is important that the tower is the last field since it is dynamically sized. The key,
/// reference count, and height are kept close to the tower to improve cache locality during
/// skip list traversal.
#[repr(C)]
pub struct Node<K, V> {
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

    #[inline]
    unsafe fn tower(&self, level: usize) -> &Atomic<Self> {
        self.pointers.get_unchecked(level)
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

    /// Marks all pointers in the tower and returns `true` if the level 0 wasn't already marked.
    fn mark_tower(&self) -> bool {
        let height = self.height();

        for level in (0..height).rev() {
            let next =
                unsafe { self.tower(level).fetch_or(1, SeqCst, epoch::unprotected()) };

            if level == 0 && next.tag() == 1 {
                return false;
            }
        }

        true
    }

    fn is_removed(&self) -> bool {
        unsafe { self.tower(0).load(SeqCst, epoch::unprotected()).tag() == 1 }
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

pub struct SkipList<K, V> {
    head: *const Node<K, V>, // !Send + !Sync
    seed: AtomicUsize,
    // TODO: Embed a custom `crossbeam_epoch::Collector` here. If `needs_drop::<K>()`, create a
    // custom collector, otherwise use the default one. Then we can remove the `K: 'static` bound.
}

unsafe impl<K: Send + Sync, V: Send + Sync> Send for SkipList<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for SkipList<K, V> {}

impl<K, V> SkipList<K, V> {
    /// Returns a new, empty skip list.
    pub fn new() -> SkipList<K, V> {
        SkipList {
            head: unsafe { Node::alloc(MAX_HEIGHT) },
            seed: AtomicUsize::new(1),
        }
    }
}

impl<K, V> SkipList<K, V>
where
    K: Ord + Send + 'static,
{
    /// Returns `true` if the skip list is empty.
    pub fn is_empty(&self) -> bool {
        unsafe {
            if (*self.head).tower(0).load(SeqCst, epoch::unprotected()).is_null() {
                return true;
            }

            let guard = &epoch::pin();

            loop {
                let head = &*self.head;
                let ptr = head.tower(0).load(SeqCst, guard);

                match ptr.as_ref() {
                    None => return true,
                    Some(n) => {
                        if n.is_removed() {
                            self.search(Some(&n.key), guard);
                        } else {
                            return false;
                        }
                    }
                }
            }
        }
    }

    /// Iterates over the skip list and returns the number of traversed elements.
    pub fn count(&self) -> usize {
        let guard = &mut epoch::pin();
        let mut count = 0;

        if let Some(mut entry) = self.front() {
            loop {
                count += 1;
                if count % 128 == 0 {
                    guard.repin();
                }

                if !entry.next() {
                    break;
                }
            }
        }

        count
    }

    /// Returns the entry with the smallest key.
    pub fn front(&self) -> Option<Entry<K, V>> {
        let guard = &epoch::pin();

        loop {
            unsafe {
                let head = &*self.head;
                let ptr = head.tower(0).load(SeqCst, guard);

                match ptr.as_ref() {
                    None => return None,
                    Some(n) => {
                        if n.is_removed() {
                            self.search(Some(&n.key), guard);
                        } else if Node::increment(n) {
                            return Some(Entry::from_raw(self, n));
                        }
                    }
                }
            }
        }
    }

    /// Returns the entry with the largest key.
    pub fn back(&self) -> Option<Entry<K, V>> {
        let guard = &epoch::pin();

        loop {
            let search = self.search(None, guard);
            let node = search.left[0];

            unsafe {
                if ptr::eq(node, self.head) {
                    return None;
                } else if Node::increment(node) {
                    return Some(Entry::from_raw(self, node));
                }
            }
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

        let mut height = num.trailing_zeros() as usize + 1;
        unsafe {
            let guard = epoch::unprotected();
            while height >= 4 && (*self.head).tower(height - 2).load(Relaxed, guard).is_null() {
                height -= 1;
            }
        }
        height
    }

    fn search<'g, Q>(&self, key: Option<&Q>, guard: &'g Guard) -> Search<'g, K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        unsafe {
            let mut s = Search {
                found: false,
                left: mem::uninitialized(),
                right: mem::uninitialized(),
            };

            'search: loop {
                let mut level = MAX_HEIGHT;

                while level >= 1 && (*self.head).tower(level - 1).load(SeqCst, guard).is_null() {
                    level -= 1;

                    s.left[level] = &*self.head;
                    s.right[level] = Shared::null();
                }

                // The current node we're at.
                let mut node = &*self.head;

                // Traverse the skip list from the highest to the lowest level.
                while level >= 1 {
                    level -= 1;

                    let mut pred = node;
                    let mut curr = pred.tower(level).load(SeqCst, guard);

                    // If `curr` is marked, that means `pred` is deleted and we have to restart the
                    // search.
                    if curr.tag() == 1 {
                        continue 'search;
                    }

                    // Iterate through the current level until we reach a node with a key greater
                    // than or equal to `key`.
                    while let Some(c) = curr.as_ref() {
                        let succ = c.tower(level).load(SeqCst, guard);

                        if succ.tag() == 1 {
                            // If `succ` is marked, that means `curr` is deleted. Let's try
                            // unlinking it from the skip list at this level.
                            match pred.tower(level).compare_and_set(
                                curr,
                                succ.with_tag(0),
                                SeqCst,
                                guard,
                            ) {
                                Ok(_) => {
                                    // On success, decrement the reference counter of `curr` and
                                    // continue searching through the current level.
                                    Node::decrement(curr.as_raw());
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

                        // If `curr` contains a key that is greater than or equal to `key`, we're
                        // done with this level.
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
                s.found = s.right[0].as_ref().map(|r| Some(r.key.borrow()) == key) == Some(true);

                return s;
            }
        }
    }

    pub fn insert(&self, key: K, value: V, replace: bool) -> Entry<K, V> {
        let guard = &epoch::pin();
        let mut search;

        unsafe {
            loop {
                // First try searching for the key.
                // Note that the `Ord` implementation for `K` may panic during the search.
                search = self.search(Some(&key), guard);

                if !search.found {
                    break;
                }

                // If a node with the key was found and we're not going to replace it, let's try
                // creating an entry positioned to it.
                let r = search.right[0];

                if replace {
                    r.deref().mark_tower();
                } else {
                    // Try incrementing its reference count.
                    if Node::increment(r.as_raw()) {
                        // Success!
                        return Entry::from_raw(self, r.as_raw());
                    }
                    // If we couldn't increment the reference count, that means that someone has
                    // just deleted the node.
                    break;
                }
            }

            // Create a new node.
            let height = self.random_height();

            let (node, n) = {
                let n = Node::<K, V>::alloc(height);

                // Write the key and the value into the node.
                ptr::write(&mut (*n).key, key);
                ptr::write(&mut (*n).value, value);

                // The reference count is initially zero. Let's increment it twice to account for:
                // 1. The entry that will be returned.
                // 2. The link at the level 0 of the tower.
                (*n).refs_and_height.fetch_add(2 << HEIGHT_BITS, Relaxed);

                (Shared::<Node<K, V>>::from(n as *const _), &*n)
            };

            loop {
                // Set the lowest successor of `n` to `search.right[0]`.
                n.tower(0).store(search.right[0], SeqCst);

                // Try installing the new node into the skip list.
                if search.left[0].tower(0)
                    .compare_and_set(search.right[0], node, SeqCst, guard)
                    .is_ok()
                {
                    break;
                }

                // We failed. Let's search for the key and try again.
                search = {
                    // Create a guard that destroys the new node in case search panics.
                    defer_on_unwind! {{
                        ptr::drop_in_place(&n.key as *const K as *mut K);
                        ptr::drop_in_place(&n.value as *const V as *mut V);
                        Node::dealloc(node.as_raw() as *mut Node<K, V>);
                    }}
                    self.search(Some(&n.key), guard)
                };

                // If a node with the key was found and we're not going to replace it, let's try
                // creating an entry positioned to it.
                if search.found {
                    let r = search.right[0];

                    if replace {
                        r.deref().mark_tower();
                    } else {
                        // Try incrementing its reference count.
                        if Node::increment(r.as_raw()) {
                            // Success! Let's deallocate the new node and return an entry
                            // positioned to the existing one.
                            let raw = node.as_raw() as *mut Node<K, V>;
                            ptr::drop_in_place(&mut (*raw).key);
                            ptr::drop_in_place(&mut (*raw).value);
                            Node::dealloc(raw);

                            return Entry::from_raw(self, r.as_raw());
                        }
                        // If we couldn't increment the reference count, that means that someone
                        // has just deleted the node.
                    }
                }
            }

            // The node was successfully inserted. Let's create an entry positioned to it.
            let entry = Entry::from_raw(self, n);

            // Build the rest of the tower above level 0.
            'build: for level in 1..height {
                loop {
                    // Obtain the predecessor and successor at the current level.
                    let pred = search.left[level];
                    let succ = search.right[level];

                    // Load the current value of the pointer in the tower.
                    let next = n.tower(level).load(SeqCst, guard);

                    // If the current pointer is marked, that means another thread is already
                    // deleting the node we've just inserted. In that case, let's just stop
                    // building the tower.
                    if next.tag() == 1 {
                        break 'build;
                    }

                    // When searching for `key` and traversing the skip list from the highest level
                    // to the lowest, it is possible to observe a node with an equal key at higher
                    // levels and then find it missing at the lower levels if it gets removed
                    // during traversal.  Even worse, it is possible to observe completely
                    // different nodes with the exact same key at different levels.
                    //
                    // Linking the new node to a dead successor with an equal key would create
                    // subtle corner cases that would require special care. It's much easier to
                    // simply prevent linking two nodes with equal keys.
                    //
                    // If the successor has the same key as the new node, that means it is marked
                    // as deleted and should be unlinked from the skip list. In that case, let's
                    // repeat the search to make sure it gets unlinked and try again.
                    if succ.as_ref().map(|s| &s.key) == Some(&n.key) {
                        // If this search panics, we simply stop building the tower without
                        // breaking any invariants. Note that building higher levels is completely
                        // optional.  Only the lowest level really matters, and all the higher
                        // levels are there just to make searching faster.
                        search = self.search(Some(&n.key), guard);
                        continue;
                    }

                    // Change the pointer at the current level from `next` to `succ`. If this CAS
                    // operation fails, that means another thread has marked the pointer and we
                    // should stop building the tower.
                    if n.tower(level)
                        .compare_and_set(next, succ, SeqCst, guard)
                        .is_err()
                    {
                        break 'build;
                    }

                    // Increment the reference count. The current value will always be at least 1
                    // because we are holding
                    (*n).refs_and_height.fetch_add(1 << HEIGHT_BITS, Relaxed);

                    // Try installing the new node at the current level.
                    if pred.tower(level)
                        .compare_and_set(succ, node, SeqCst, guard)
                        .is_ok()
                    {
                        // Success! Continue on the next level.
                        break;
                    }

                    // Installation failed. Decrement the reference count.
                    (*n).refs_and_height.fetch_sub(1 << HEIGHT_BITS, Relaxed);

                    // We don't have the most up-to-date search results. Repeat the search.
                    //
                    // If this search panics, we simply stop building the tower without breaking
                    // any invariants. Note that building higher levels is completely optional.
                    // Only the lowest level really matters, and all the higher levels are there
                    // just to make searching faster.
                    search = self.search(Some(&n.key), guard);
                }
            }

            // If a pointer in the tower is marked, that means our node is in the process of
            // deletion or already deleted. It is possible that another thread (either partially or
            // completely) deleted the new node while we were building the tower, and just after
            // that we installed the new node at one of the higher levels. In order to undo that
            // installation, we must repeat the search, which will unlink the new node at that
            // level.
            if n.tower(height - 1).load(SeqCst, guard).tag() == 1 {
                self.search(Some(&n.key), guard);
            }

            entry
        }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<Entry<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let guard = &epoch::pin();

        loop {
            let search = self.search(Some(key), guard);
            if !search.found {
                return None;
            }

            let node = search.right[0].as_raw();

            unsafe {
                if Node::increment(node) {
                    return Some(Entry::from_raw(self, node));
                }
            }
        }
    }

    pub fn remove<Q>(&self, key: &Q) -> Option<Entry<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let guard = &epoch::pin();

        loop {
            // Try searching for the key.
            let search = self.search(Some(key), guard);
            if !search.found {
                return None;
            }

            let node = search.right[0];

            unsafe {
                let n = node.deref();

                if !Node::increment(n) {
                    continue;
                }

                let entry = Entry::from_raw(self, n);

                if !n.mark_tower() {
                    continue;
                }

                for level in (0..n.height()).rev() {
                    let succ = n.tower(level).load(SeqCst, guard).with_tag(0);

                    if search.left[level].tower(level)
                        .compare_and_set(node, succ, SeqCst, guard)
                        .is_ok()
                    {
                        Node::decrement(n);
                    } else {
                        self.search(Some(key), guard);
                        break;
                    }
                }

                return Some(entry);
            }
        }
    }

    pub fn clear(&self) {
        let guard = &mut epoch::pin();

        let mut count = 0;

        if let Some(mut entry) = self.front() {
            loop {
                count += 1;
                if count % 128 == 0 {
                    guard.repin();
                }

                let next = entry.get_next();
                entry.node.mark_tower();

                match next {
                    None => break,
                    Some(e) => entry = e,
                }
            }

        }

        self.search(None, guard);
    }
}

impl<K, V> Drop for SkipList<K, V> {
    fn drop(&mut self) {
        let mut node = self.head as *mut Node<K, V>;

        while !node.is_null() {
            unsafe {
                if node as *const _ != self.head {
                    ptr::drop_in_place(&mut (*node).key);
                    ptr::drop_in_place(&mut (*node).value);
                }

                let next = (*node).tower(0).load(Relaxed, epoch::unprotected());
                Node::dealloc(node);
                node = next.as_raw() as *mut Node<K, V>;
            }
        }
    }
}

impl<K, V> IntoIterator for SkipList<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        unsafe {
            let next = (*self.head).tower(0)
                .load(Relaxed, epoch::unprotected())
                .as_raw();

            for level in 0..MAX_HEIGHT {
                (*self.head).tower(level).store(Shared::null(), Relaxed);
            }

            IntoIter {
                node: next as *mut Node<K, V>,
            }
        }
    }
}

/// A search result.
///
/// The result indicates whether the key was found, as well as what were the adjacent nodes to the
/// key on each level of the skip list.
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

pub struct Entry<'a, K, V>
where
    K: Send + 'static,
    V: 'a,
{
    pub parent: &'a SkipList<K, V>,
    pub node: &'a Node<K, V>,
}

unsafe impl<'a, K: Send + Sync, V: Send + Sync> Send for Entry<'a, K, V> {}
unsafe impl<'a, K: Send + Sync, V: Send + Sync> Sync for Entry<'a, K, V> {}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Send + 'static,
{
    fn new(parent: &'a SkipList<K, V>, node: &'a Node<K, V>) -> Self {
        Entry {
            parent,
            node,
        }
    }

    unsafe fn from_raw(parent: &'a SkipList<K, V>, node: *const Node<K, V>) -> Self {
        Entry {
            parent,
            node: &*node,
        }
    }

    /// Returns `true` if this entry is removed from the skip list.
    pub fn is_removed(&self) -> bool {
        self.node.is_removed()
    }

    /// Returns a reference to the key.
    pub fn key(&self) -> &K {
        &self.node.key
    }

    /// Returns a reference to the value.
    pub fn value(&self) -> &V {
        &self.node.value
    }
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Ord + Send + 'static,
{
    /// Moves to the next entry in the skip list.
    pub fn next(&mut self) -> bool {
        match self.get_next() {
            None => false,
            Some(n) => {
                *self = n;
                true
            }
        }
    }

    /// Moves to the previous entry in the skip list.
    pub fn prev(&mut self) -> bool {
        match self.get_prev() {
            None => false,
            Some(n) => {
                *self = n;
                true
            }
        }
    }

    pub fn get_next(&self) -> Option<Entry<'a, K, V>> {
        let guard = &epoch::pin();

        loop {
            unsafe {
                let succ = {
                    let succ = self.node.tower(0).load(SeqCst, guard);
                    if succ.tag() == 0 {
                        succ
                    } else {
                        let search = self.parent.search(Some(&self.node.key), guard);
                        if search.found {
                            search.right[0].deref().tower(0).load(SeqCst, guard)
                        } else {
                            search.right[0]
                        }
                    }
                };

                if succ.tag() == 0 {
                    match succ.as_ref() {
                        None => return None,
                        Some(s) => {
                            if !s.is_removed() && Node::increment(s) {
                                return Some(Entry::from_raw(self.parent, s));
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn get_prev(&self) -> Option<Entry<'a, K, V>> {
        let guard = &epoch::pin();

        loop {
            let search = self.parent.search(Some(self.key()), guard);
            let pred = search.left[0];

            if ptr::eq(pred, self.parent.head) {
                return None;
            }

            unsafe {
                if Node::increment(pred) {
                    return Some(Entry::from_raw(self.parent, pred));
                }
            }
        }
    }

    /// Removes this entry from the skip list.
    ///
    /// Returns `true` if this call removed the entry and `false` if it was already removed.
    pub fn remove(&self) -> bool {
        if self.node.mark_tower() {
            let guard = &epoch::pin();
            self.parent.search(Some(&self.node.key), guard);
            true
        } else {
            false
        }
    }
}

impl<'a, K, V> Drop for Entry<'a, K, V>
where
    K: Send + 'static,
{
    fn drop(&mut self) {
        unsafe { Node::decrement(self.node) }
    }
}

impl<'a, K, V> Clone for Entry<'a, K, V>
where
    K: Send + 'static,
{
    fn clone(&self) -> Entry<'a, K, V> {
        unsafe {
            Node::increment(self.node);
        }
        Entry {
            parent: self.parent,
            node: self.node,
        }
    }
}

impl<'a, K, V> fmt::Debug for Entry<'a, K, V>
where
    K: Send + fmt::Debug + 'static,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Entry")
            .field(&self.key())
            .field(&self.value())
            .finish()
    }
}

pub struct IntoIter<K, V> {
    node: *mut Node<K, V>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        loop {
            if self.node.is_null() {
                return None;
            }

            unsafe {
                let key = ptr::read(&mut (*self.node).key);
                let value = ptr::read(&mut (*self.node).value);

                let next = (*self.node).tower(0).load(Relaxed, epoch::unprotected());
                self.node = next.as_raw() as *mut Node<K, V>;

                if next.tag() == 0 {
                    return Some((key, value));
                }
            }
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

    // TODO: random panics
    // TODO: stress test
    // TODO: https://github.com/dmlloyd/openjdk/blob/jdk/jdk/test/jdk/java/util/concurrent/tck/ConcurrentSkipListMapTest.java

    #[test]
    fn new() {
        SkipList::<i32, i32>::new();
        SkipList::<String, Box<i32>>::new();
    }

    #[test]
    fn is_empty() {
        let s = SkipList::new();
        assert!(s.is_empty());

        s.insert(1, 10, false);
        assert!(!s.is_empty());
        s.insert(2, 20, false);
        s.insert(3, 30, false);
        assert!(!s.is_empty());

        s.remove(&2);
        assert!(!s.is_empty());

        s.remove(&1);
        assert!(!s.is_empty());

        s.remove(&3);
        assert!(s.is_empty());
    }

    #[test]
    fn insert() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5];
        let not_present = [1, 3, 6, 9, 10];
        let s = SkipList::new();

        for &elt in &insert {
            s.insert(elt, elt * 10, false);
            assert_eq!(*s.get(&elt).unwrap().value(), elt * 10);
        }

        for &elt in &not_present {
            assert!(s.get(&elt).is_none());
        }
    }

    #[test]
    fn remove() {
        let insert = [0, 4, 2, 12, 8, 7, 11, 5];
        let not_present = [1, 3, 6, 9, 10];
        let remove = [2, 12, 8];

        let s = SkipList::new();

        for &elt in &insert {
            s.insert(elt, elt * 10, false);
        }

        for elt in &not_present {
            assert!(s.remove(elt).is_none());
        }

        for elt in &remove {
            assert!(!s.remove(elt).is_none());
        }

        let mut v = vec![];
        let mut e = s.front().unwrap();
        loop {
            v.push(*e.key());
            if !e.next() {
                break;
            }
        }
        assert_eq!(v, [0, 4, 5, 7, 11]);

        for elt in &insert {
            s.remove(elt);
        }

        assert!(s.is_empty());
    }

    #[test]
    fn entry() {
        let insert = [4, 2, 12, 8, 7, 11, 5];
        let s = SkipList::new();

        assert!(s.front().is_none());
        assert!(s.back().is_none());

        for &elt in &insert {
            s.insert(elt, elt * 10, false);
        }

        let mut e = s.front().unwrap();
        assert_eq!(*e.key(), 2);
        assert!(!e.prev());
        assert!(e.next());
        assert_eq!(*e.key(), 4);

        e = s.back().unwrap();
        assert_eq!(*e.key(), 12);
        assert!(!e.next());
        assert!(e.prev());
        assert_eq!(*e.key(), 11);
    }

    #[test]
    fn entry_remove() {
        let insert = [4, 2, 12, 8, 7, 11, 5];
        let s = SkipList::new();
        for &elt in &insert {
            s.insert(elt, elt * 10, false);
        }

        let mut e = s.get(&7).unwrap();
        assert!(!e.is_removed());
        assert!(e.remove());
        assert!(e.is_removed());

        e.prev();
        e.next();
        assert_ne!(*e.key(), 7);
    }

    #[test]
    fn entry_reposition() {
        let insert = [4, 2, 12, 8, 7, 11, 5];
        let s = SkipList::new();
        for &elt in &insert {
            s.insert(elt, elt * 10, false);
        }

        let mut e = s.get(&7).unwrap();
        assert!(!e.is_removed());
        assert!(e.remove());
        assert!(e.is_removed());

        s.insert(7, 700, false);
        e.prev();
        e.next();
        assert_eq!(*e.key(), 7);
    }

    #[test]
    fn count() {
        let insert = [4, 2, 12, 8, 7, 11, 5];
        let s = SkipList::new();
        assert_eq!(s.count(), 0);

        for (index, &elt) in insert.iter().enumerate() {
            s.insert(elt, elt * 10, false);
            assert_eq!(s.count(), index + 1);
        }

        s.insert(5, 0, false);
        assert_eq!(s.count(), 7);
        s.insert(5, 0, true);
        assert_eq!(s.count(), 7);

        s.remove(&6);
        assert_eq!(s.count(), 7);
        s.remove(&5);
        assert_eq!(s.count(), 6);
        s.remove(&12);
        assert_eq!(s.count(), 5);
    }
}
