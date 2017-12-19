use std::borrow::Borrow;
use std::fmt;

use base;

pub struct SkipMap<K, V> {
    inner: base::SkipList<K, V>,
}

impl<K, V> SkipMap<K, V>
where
    K: Ord + Send + 'static,
{
    pub fn new() -> SkipMap<K, V> {
        SkipMap {
            inner: base::SkipList::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn count(&self) -> usize {
        self.inner.count()
    }

    pub fn cursor(&self) -> Cursor<K, V> {
        Cursor {
            inner: self.inner.cursor(),
        }
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
        Cursor {
            inner: self.inner.get(key),
        }
    }

    pub fn seek<Q>(&self, key: &Q) -> Cursor<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut cursor = self.cursor();
        cursor.seek(key);
        cursor
    }

    pub fn insert(&self, key: K, value: V) -> Cursor<K, V> {
        unimplemented!()
    }

    fn get_or_insert(&self, key: K, value: V) -> Cursor<K, V> {
        unimplemented!()
    }

    pub fn remove<Q>(&self, key: &Q) -> Cursor<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        Cursor {
            inner: self.inner.remove(key),
        }
    }

    pub fn clear(&self) {
        let mut cursor = self.cursor();
        cursor.next();

        while !cursor.is_null() {
            cursor.remove();
            cursor.next();
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        let mut cursor = self.cursor();
        cursor.next();

        while !cursor.is_null() {
            if f(cursor.key().unwrap(), cursor.value().unwrap()) {
                cursor.remove();
            }
            cursor.next();
        }
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

impl<K, V> fmt::Debug for SkipMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SkipMap").finish()
    }
}

// TODO: Cursor::set(V) where K: Clone (what if null? ignore?)
// TODO: Cursor::replace(V) -> Option<Cursor<K, V>> where K: Clone, must be atomic (what if null? ignore? what if removed?)
// TODO: Cursor::reload() (if it's removed, searches again)

pub struct Cursor<'a, K, V>
where
    K: Send + 'static,
    V: 'a,
{
    inner: base::Cursor<'a, K, V>,
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
