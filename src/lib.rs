extern crate crossbeam_epoch as epoch;
extern crate crossbeam_utils as utils;
#[macro_use]
extern crate scopeguard;

mod base;
pub mod map;
pub mod set;

// TODO: pub mod range; // RangeArgument

pub use map::SkipMap;
pub use set::SkipSet;

// TODO: Heap, impl as SkipListMap<K, ManuallyDrop<V>>
// TODO: Entry::try_into_value()?
