extern crate crossbeam_epoch as epoch;
extern crate crossbeam_utils as utils;
#[macro_use]
extern crate scopeguard;

pub mod map;
mod base;

// TODO: pub mod range; // RangeArgument

pub use map::SkipMap;

// TODO: Heap, impl as SkipListMap<K, ManuallyDrop<V>>
