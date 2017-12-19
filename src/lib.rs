extern crate crossbeam_epoch as epoch;
#[macro_use]
extern crate scopeguard;

pub mod map;
mod base;

// TODO: pub mod queue; // SkipQueue

pub use map::SkipMap;

// TODO: Heap, impl as SkipListMap<K, ManuallyDrop<V>>
