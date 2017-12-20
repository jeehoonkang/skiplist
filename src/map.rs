use std::borrow::Borrow;
use std::fmt;
use std::iter::FromIterator;

use base;

pub struct SkipMap<K, V> {
    inner: base::SkipList<K, V>,
}

impl<K, V> SkipMap<K, V> {
    /// Returns a new, empty map.
    pub fn new() -> SkipMap<K, V> {
        SkipMap {
            inner: base::SkipList::new(),
        }
    }
}

impl<K, V> SkipMap<K, V>
where
    K: Ord + Send + 'static,
{
    /// Returns `true` if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterates over the skip list and returns the number of traversed elements.
    pub fn count(&self) -> usize {
        self.inner.count()
    }

    /// Returns a cursor positioned to null.
    pub fn cursor(&self) -> Cursor<K, V> {
        Cursor {
            inner: self.inner.cursor(),
        }
    }

    /// Returns a cursor positioned to the first element in the map.
    ///
    /// If the map is empty, the cursor will be positioned to null.
    pub fn front(&self) -> Cursor<K, V> {
        let cursor = self.cursor();
        cursor.seek_to_front();
        cursor
    }

    /// Returns a cursor positioned to the last element in the map.
    ///
    /// If the map is empty, the cursor will be positioned to null.
    pub fn back(&self) -> Cursor<K, V> {
        let cursor = self.cursor();
        cursor.seek_to_back();
        cursor
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        !self.get(key).is_null()
    }

    /// Returns a cursor positioned to the element corresponding to the key.
    ///
    /// If the such an element doesn't exist, the cursor will be positioned to null.
    pub fn get<Q>(&self, key: &Q) -> Cursor<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        Cursor {
            inner: self.inner.get(key),
        }
    }

    /// Seeks the first element with key equal to or greater than `key`.
    ///
    /// The returned cursor will be positioned to the found element, or null if the map is empty.
    pub fn seek<Q>(&self, key: &Q) -> Cursor<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let cursor = self.cursor();
        cursor.seek(key);
        cursor
    }

    /// Inserts a new key-value pair into the map.
    ///
    /// If there is an existing pair with this key, it will be removed before inserting the new
    /// pair. The returned cursor will be positioned to the new pair.
    pub fn insert(&self, key: K, value: V) -> Cursor<K, V> {
        Cursor {
            inner: self.inner.insert(key, value, true),
        }
    }

    /// Finds an element with the specified key, or inserts a new key-value pair if it doesn't
    /// exist.
    ///
    /// The returned cursor will be positioned to the found element, or the new one if it was
    /// inserted.
    fn get_or_insert(&self, key: K, value: V) -> Cursor<K, V> {
        Cursor {
            inner: self.inner.insert(key, value, false),
        }
    }

    /// Removes an element from the map and returns a cursor positioned to it.
    ///
    /// If no element with the specified key exists, the returned cursor will be positioned to
    /// null.
    pub fn remove<Q>(&self, key: &Q) -> Cursor<K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        Cursor {
            inner: self.inner.remove(key),
        }
    }

    /// Clears the map, removing all elements.
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Retains only the elements for which the predicate function returns `true`.
    ///
    /// In other words, remove all pairs `(k, v)` such that `f(&k, &v)` returns `false`.
    pub fn retain<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        let cursor = self.cursor();
        cursor.next();

        loop {
            match cursor.key_and_value() {
                None => break,
                Some((k, v)) => {
                    if f(k, v) {
                        cursor.remove();
                    }
                }
            }
            cursor.next();
        }
    }

    // TODO:
    // pub fn drain<Q, R>(&self, range: R) -> Drain<K, V>
    // where
    //     K: Borrow<Q>,
    //     Q: Ord + ?Sized,
    //     R: RangeArgument<Q>,
    // {
    //     unimplemented!()
    // }

    // TODO:
    // 1. `fn iter(&self) -> Iter<K, V>` -> Entry<'a, K, V> -> deref to (&'a K, &'a V).
    // 4. `fn range<Q, R>(&self, range: R) -> Range<K, V> where ...` -> Entry<'a, K, V> (double
    //    ended iterator).
}

impl<K, V> Default for SkipMap<K, V> {
    fn default() -> SkipMap<K, V> {
        SkipMap::new()
    }
}

impl<K, V> fmt::Debug for SkipMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO(stjepang): Iterate over all elements (see `BTreeMap`).
        f.debug_struct("SkipMap").finish()
    }
}

impl<K, V> IntoIterator for SkipMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter {
            inner: self.inner.into_iter(),
        }
    }
}

impl<K, V> FromIterator<(K, V)> for SkipMap<K, V>
where
    K: Ord + Send + 'static,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> SkipMap<K, V> {
        let s = SkipMap::new();
        for (k, v) in iter {
            s.insert(k, v);
        }
        s
    }
}

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

    /// Returns the key of the current element.
    pub fn key(&self) -> Option<&K> {
        self.inner.key()
    }

    /// Returns the value of the current element.
    pub fn value(&self) -> Option<&V> {
        self.inner.value()
    }

    /// Returns the key-value pair of the current element.
    pub fn key_and_value(&self) -> Option<(&K, &V)> {
        self.inner.key_and_value()
    }
}

impl<'a, K, V> Cursor<'a, K, V>
where
    K: Ord + Send + 'static,
{
    /// Moves the cursor to the next element in the skip list.
    pub fn next(&self) {
        self.inner.next();
    }

    /// Moves the cursor to the previous element in the skip list.
    pub fn prev(&self) {
        self.inner.prev();
    }

    /// Positions the cursor to the first element with key equal to or greater than `key`.
    ///
    /// Returns `true` if an element with `key` is found.
    pub fn seek<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.seek(key)
    }

    /// Positions the cursor to the first element in the skip list, if it exists.
    pub fn seek_to_front(&self) -> bool {
        self.inner.seek_to_front()
    }

    /// Positions the cursor to the last element in the skip list, if it exists.
    pub fn seek_to_back(&self) -> bool {
        self.inner.seek_to_back()
    }

    pub fn seek_to_null(&self) {
        // TODO
        unimplemented!()
    }

    /// If the current element is removed, seeks for its key to reposition the cursor.
    ///
    /// Returns `true` if the cursor didn't need repositioning or if the key didn't change after
    /// repositioning.
    ///
    /// Otherwise, `false` is returned and the cursor is positioned to the first element with a
    /// greater key. If no element with a greater key exists, then the cursor is positioned to
    /// null.
    pub fn reseek(&self) -> bool {
        self.inner.reseek()
    }

    /// Inserts a new key-value pair into the map.
    ///
    /// The cursor will be positioned to the new pair and the old one
    /// If there is an existing pair with this key, it will be removed before inserting the new
    /// pair. The returned cursor will be positioned to the new pair.
    pub fn insert(&self, value: V) -> Cursor<K, V>
    where
        K: Clone,
    {
        match self.key() {
            None => self.clone(),
            Some(k) => {
                let c = self.inner.parent.insert(k.clone(), value, true);
                self.inner.node.swap(&c.node);
                Cursor {
                    inner: c,
                }
            }
        }
    }

    /// Removes the element this cursor is positioned to.
    ///
    /// Returns `true` if this call removed the element and `false` if it was already removed.
    pub fn remove(&self) -> bool {
        self.inner.remove()
    }
}

impl<'a, K, V> Clone for Cursor<'a, K, V>
where
    K: Send + 'static,
{
    fn clone(&self) -> Cursor<'a, K, V> {
        Cursor {
            inner: self.inner.clone(),
        }
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

pub struct IntoIter<K, V> {
    inner: base::IntoIter<K, V>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        self.inner.next()
    }
}

impl<K, V> fmt::Debug for IntoIter<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO(stjepang): Iterate over all elements (see `btree_map::IntoIter`).
        f.debug_struct("IntoIter").finish()
    }
}

#[cfg(test)]
mod tests {}
