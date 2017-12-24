use std::borrow::Borrow;
use std::fmt;
use std::iter::FromIterator;

use base;

// TODO: len()

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

    /// Iterates over the map and returns the number of traversed elements.
    pub fn count(&self) -> usize {
        self.inner.count()
    }

    /// Returns the entry with the smallest key.
    pub fn front(&self) -> Option<Entry<K, V>> {
        let c = self.inner.cursor();
        c.seek_to_front();

        if c.is_null() {
            None
        } else {
            Some(Entry { inner: c })
        }
    }

    /// Returns the entry with the largest key.
    pub fn back(&self) -> Option<Entry<K, V>> {
        let c = self.inner.cursor();
        c.seek_to_back();

        if c.is_null() {
            None
        } else {
            Some(Entry { inner: c })
        }
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Returns a cursor positioned to the element corresponding to the key.
    ///
    /// If the such an element doesn't exist, the cursor will be positioned to null.
    pub fn get<Q>(&self, key: &Q) -> Option<Entry<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let c = self.inner.get(key);
        if c.is_null() {
            None
        } else {
            Some(Entry { inner: c })
        }
    }

    /// Inserts a new key-value pair into the map.
    ///
    /// If there is an existing pair with this key, it will be removed before inserting the new
    /// pair. The returned cursor will be positioned to the new pair.
    pub fn insert(&self, key: K, value: V) -> Entry<K, V> {
        Entry {
            inner: self.inner.insert(key, value, true),
        }
    }

    /// Finds an element with the specified key, or inserts a new key-value pair if it doesn't
    /// exist.
    ///
    /// The returned cursor will be positioned to the found element, or the new one if it was
    /// inserted.
    pub fn get_or_insert(&self, key: K, value: V) -> Entry<K, V> {
        Entry {
            inner: self.inner.insert(key, value, false),
        }
    }

    /// Removes an element from the map and returns a cursor positioned to it.
    ///
    /// If no element with the specified key exists, the returned cursor will be positioned to
    /// null.
    pub fn remove<Q>(&self, key: &Q) -> Option<Entry<K, V>>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let c = self.inner.remove(key);

        if c.is_null() {
            None
        } else {
            Some(Entry { inner: c })
        }
    }

    /// Clears the map, removing all elements.
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// TODO
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            parent: self,
            entry: None,
            finished: false,
        }
    }

    // TODO:
    // `pub fn range<Q, R>(&self, range: R) -> Range<K, V> where ...` -> Entry<'a, K, V> (double ended iterator).
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

impl<'a, K, V> IntoIterator for &'a SkipMap<K, V>
where
    K: Ord + Send + 'static,
{
    type Item = Entry<'a, K, V>;
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
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

pub struct Entry<'a, K, V>
where
    K: Send + 'static,
    V: 'a,
{
    inner: base::Cursor<'a, K, V>,
}

unsafe impl<'a, K: Send + Sync, V: Send + Sync> Send for Entry<'a, K, V> {}
unsafe impl<'a, K: Send + Sync, V: Send + Sync> Sync for Entry<'a, K, V> {}

// TODO: rename 'cursor' in docs

impl<'a, K, V> Entry<'a, K, V>
where
    K: Send + 'static,
{
    /// Returns the key of the current element.
    pub fn key(&self) -> &K {
        self.inner.key().unwrap()
    }

    /// Returns the value of the current element.
    pub fn value(&self) -> &V {
        self.inner.value().unwrap()
    }

    /// Returns `true` if the cursor is positioned to a valid element (not null and not removed).
    pub fn is_removed(&self) -> bool {
        // self.inner.is_removed()
        unimplemented!()
    }
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Ord + Send + 'static,
{
    /// Moves the cursor to the next element in the map.
    pub fn next(&self) -> Option<Entry<'a, K, V>> {
        let c = self.inner.clone();
        c.next();

        if c.is_null() {
            None
        } else {
            Some(Entry { inner: c })
        }
    }

    /// Moves the cursor to the previous element in the map.
    pub fn prev(&self) -> Option<Entry<'a, K, V>> {
        // self.inner.prev();
        unimplemented!()
    }

    /// Removes the element this cursor is positioned to.
    ///
    /// Returns `true` if this call removed the element and `false` if it was already removed.
    pub fn remove(&self) -> bool {
        self.inner.remove()
    }
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Clone + Ord + Send + 'static,
{
    /// Inserts a new key-value pair into the map. TODO
    ///
    /// The cursor will be positioned to the new pair and the old one
    /// If there is an existing pair with this key, it will be removed before inserting the new
    /// pair. The returned cursor will be positioned to the new pair.
    pub fn replace(&self, value: V) -> Option<Entry<'a, K, V>> {
        // match self.key() {
        //     None => self.clone(),
        //     Some(k) => {
        //         let c = self.inner.parent.insert(k.clone(), value, true);
        //         self.inner.node.swap(&c.node);
        //         Entry {
        //             inner: c,
        //         }
        //     }
        // }
        unimplemented!()
    }
}

impl<'a, K, V> Clone for Entry<'a, K, V>
where
    K: Send + 'static,
{
    fn clone(&self) -> Entry<'a, K, V> {
        Entry {
            inner: self.inner.clone(),
        }
    }
}

// TODO: Eq, Ord for Entry?

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

pub struct Iter<'a, K, V>
where
    K: Send + 'static,
    V: 'a,
{
    parent: &'a SkipMap<K, V>,
    entry: Option<Entry<'a, K, V>>,
    finished: bool,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: Ord + Send + 'static,
    V: 'a,
{
    type Item = Entry<'a, K, V>;

    fn next(&mut self) -> Option<Entry<'a, K, V>> {
        if self.finished {
            None
        } else {
            let e = match self.entry.as_ref() {
                None => self.parent.front(),
                Some(ref e) => e.next(),
            };

            self.entry = e.clone();
            self.finished = e.is_none();
            e
        }
    }
}

impl<'a, K, V> fmt::Debug for Iter<'a, K, V>
where
    K: Send + fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO(stjepang): Iterate over all elements (see `btree_map::Iter`).
        f.debug_struct("Iter").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        let m = SkipMap::new();

        m.insert(1, 10);
        m.insert(5, 50);
        m.insert(7, 70);

        for e in &m {
            println!("{:?}", e);
        }
    }
}
