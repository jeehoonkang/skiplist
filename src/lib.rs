extern crate crossbeam_epoch as epoch;
#[macro_use]
extern crate scopeguard;

pub mod map;
mod skiplist;

pub use map::SkipListMap;
