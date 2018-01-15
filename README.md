# Lock-free skip list (WIP)

This skip list aims to be a concurrent alternative to `BTreeMap` and `BTreeSet`.

### Examples

Concurrent insertion without synchronization:

```rust
extern crate crossbeam;
extern crate skiplist;

use skiplist::SkipMap;

const THREADS: usize = 2;

fn main() {
    let map = SkipMap::new();

    crossbeam::scope(|scope| {
        for i in 0..THREADS {
            let map = &map;

            scope.spawn(move || {
                let mut num = i as u64;

                for _ in 0 .. 1_000_000 / THREADS {
                    num = num.wrapping_mul(17).wrapping_add(255);
                    map.insert(num, !num);
                }
            });
        }
    });
}
```

Basic operations on the map:

```rust
// type inference lets us omit an explicit type signature (which
// would be `SkipMap<&str, &str>` in this example).
let movie_reviews = SkipMap::new();

// review some movies.
movie_reviews.insert("Office Space",       "Deals with real issues in the workplace.");
movie_reviews.insert("Pulp Fiction",       "Masterpiece.");
movie_reviews.insert("The Godfather",      "Very enjoyable.");
movie_reviews.insert("The Blues Brothers", "Eye lyked it alot.");

// check for a specific one.
if !movie_reviews.contains_key("Les Misérables") {
    println!("We've got {} reviews, but Les Misérables ain't one.",
             movie_reviews.len());
}

// oops, this review has a lot of spelling mistakes, let's delete it.
movie_reviews.remove("The Blues Brothers");

// look up the values associated with some keys.
let to_find = ["Up!", "Office Space"];
for book in &to_find {
    match movie_reviews.get(book) {
       Some(entry) => println!("{}: {}", book, entry.value()),
       None => println!("{} is unreviewed.", book)
    }
}

// iterate over everything.
for entry in &movie_reviews {
    let movie = entry.key();
    let review = entry.value();
    println!("{}: \"{}\"", movie, review);
}
```
