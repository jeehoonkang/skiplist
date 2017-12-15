use std::borrow::Borrow;
use std::fmt;

use skiplist::{self, SkipList};

pub struct SkipListMap<K, V> {
    inner: SkipList<K, V>,
}

impl<K, V> SkipListMap<K, V>
where
    K: Ord + Send + 'static,
{
    pub fn new() -> SkipListMap<K, V> {
        SkipListMap {
            inner: SkipList::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn count(&self) -> usize {
        unimplemented!()
    }

    pub fn cursor(&self) -> Cursor<K, V> {
        unimplemented!()
    }

    pub fn front(&self) -> Cursor<K, V> {
        let mut cursor = self.cursor();
        cursor.seek_to_front();
        cursor
    }

    pub fn back(&self) -> Cursor<K, V> {
        let mut cursor = self.cursor();
        cursor.seek_to_back();
        cursor
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        !self.get(key).is_null()
    }

    pub fn get<Q>(&self, key: &Q) -> Cursor<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        unimplemented!()
    }

    pub fn insert(&self, key: K, value: V) -> Result<Cursor<K, V>, InsertError<K, V>> {
        unimplemented!()
    }

    pub fn remove<Q>(&self, key: &Q) -> Cursor<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        unimplemented!()
    }

    pub fn pop_front(&self) -> Cursor<K, V> {
        unimplemented!()
    }

    pub fn pop_back(&self) -> Cursor<K, V> {
        unimplemented!()
    }

    pub fn clear(&self) {
        unimplemented!()
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        unimplemented!()
    }

    pub fn keys(&self) {
        unimplemented!()
    }

    pub fn values(&self) {
        unimplemented!()
    }

    // pub fn range<Q, R>(&self, range: R) -> Range<K, V>
    // where
    //     K: Borrow<Q>,
    //     Q: Ord + ?Sized,
    //     R: RangeArgument<Q>,
    // {
    //     unimplemented!()
    // }

    // pub fn drain<Q, R>(&self, range: R) -> Drain<K, V>
    // where
    //     K: Borrow<Q>,
    //     Q: Ord + ?Sized,
    //     R: RangeArgument<Q>,
    // {
    //     unimplemented!()
    // }

    pub fn iter(&self) {
        unimplemented!()
    }

    pub fn into_iter(&self) { // TODO: impl IntoIterator
        unimplemented!()
    }

    // TODO: impl FromIterator
    // TODO: impl Debug
    // TODO: impl Default
}

impl<K, V> fmt::Debug for SkipListMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SkipListMap").finish()
    }
}

#[derive(Debug)]
pub struct InsertError<'a, K, V>
where
    K: Send + 'static,
    V: 'a
{
    pub key: K,
    pub value: V,
    pub cursor: Cursor<'a, K, V>,
}

pub struct Cursor<'a, K, V>
where
    K: Send + 'static,
    V: 'a,
{
    inner: skiplist::Cursor<'a, K, V>,
}

unsafe impl<'a, K: Send + Sync, V: Send + Sync> Send for Cursor<'a, K, V> {}
unsafe impl<'a, K: Send + Sync, V: Send + Sync> Sync for Cursor<'a, K, V> {}

impl<'a, K, V> Cursor<'a, K, V>
where
    K: Send + 'static,
{
    /// Returns `true` if the cursor is positioned to null.
    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }

    /// Returns `true` if the cursor is positioned to a valid element (not null and not removed).
    pub fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Returns the key of the element pointed to by the cursor.
    pub fn key(&self) -> Option<&K> {
        self.inner.key()
    }

    /// Returns the value of the element pointed to by the cursor.
    pub fn value(&self) -> Option<&V> {
        self.inner.value()
    }
}

impl<'a, K, V> Cursor<'a, K, V>
where
    K: Ord + Send + 'static,
{
    // TODO: what happens if this one is removed and there is a new element with the same key?
    /// Moves the cursor to the next element in the skip list.
    pub fn next(&mut self) {
        self.inner.next();
    }

    // TODO: what happens if this one is removed and there is a new element with the same key?
    /// Moves the cursor to the previous element in the skip list.
    pub fn prev(&mut self) {
        self.inner.prev();
    }

    /// Positions the cursor to the first element with key equal to or greater than `key`.
    pub fn seek<Q>(&mut self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.seek(key)
    }

    /// Positions the cursor to the first element in the skip list, if it exists.
    pub fn seek_to_front(&mut self) -> bool {
        self.inner.seek_to_front()
    }

    /// Positions the cursor to the last element in the skip list, if it exists.
    pub fn seek_to_back(&mut self) -> bool {
        self.inner.seek_to_back()
    }

    /// Removes the element this cursor is positioned to.
    ///
    /// Returns `true` if this call removed the element and `false` if it was already removed.
    pub fn remove(&self) -> bool {
        self.inner.remove()
    }
}

impl<'a, K, V> fmt::Debug for Cursor<'a, K, V>
where
    K: Send + fmt::Debug + 'static,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Cursor")
            .field("key", &self.key())
            .field("value", &self.value())
            .finish()
    }
}

#[cfg(test)]
mod tests {

}
