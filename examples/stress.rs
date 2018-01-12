extern crate crossbeam;
extern crate skiplist;
extern crate rand;

use std::collections::HashSet;
use std::sync::{Barrier, Mutex};
use std::time::{Duration, Instant};

use skiplist::SkipMap;
use rand::{thread_rng, Rng};

const RUN_MILLIS: u64 = 2000;

fn stress_small(num_threads: usize, limit: u32) {
    println!("stress_small({}, {})", num_threads, limit);

    let map = SkipMap::new();

    crossbeam::scope(|scope| {
        for _ in 0..num_threads {
            scope.spawn(|| {
                let mut rng = thread_rng();
                let deadline = Instant::now() + Duration::from_millis(RUN_MILLIS);

                while Instant::now() < deadline {
                    for _ in 0..1000 {
                        let x = rng.gen_range(0, limit);

                        if rng.gen() {
                            map.insert(x, x);
                        } else {
                            map.remove(&x);
                        }
                    }
                }
            });
        }
    });
}

fn stress_large(num_threads: usize) {
    println!("stress_large({})", num_threads);

    let deadline = Instant::now() + Duration::from_millis(RUN_MILLIS);

    while Instant::now() < deadline {
        let mut nums: Vec<u32> = (0..50_000).collect();
        thread_rng().shuffle(&mut nums);

        let nums = Mutex::new(nums);
        let blacklist = Mutex::new(vec![]);
        let barrier = Barrier::new(num_threads);

        let map = SkipMap::new();

        crossbeam::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|| {
                    let mut initial = vec![];
                    let mut insert = vec![];
                    let mut remove = vec![];

                    {
                        let mut nums = nums.lock().unwrap();
                        let mut blacklist = blacklist.lock().unwrap();

                        for _ in 0 .. nums.len() / num_threads / 2 {
                            let x = nums.pop().unwrap();
                            initial.push(x);
                            blacklist.push(x);
                        }

                        for _ in 0 .. nums.len() / num_threads / 2 {
                            let x = nums.pop().unwrap();
                            insert.push(x);
                            remove.push(x);
                        }
                    }

                    thread_rng().shuffle(&mut insert);
                    thread_rng().shuffle(&mut remove);

                    barrier.wait();

                    for &x in &initial {
                        map.insert(x, x);
                    }

                    for (x, y) in insert.into_iter().zip(remove.into_iter()) {
                        map.insert(x, x);
                        map.remove(&y);
                    }

                    for x in &initial {
                        map.remove(x);
                    }
                });
            }
        });

        let remaining: Vec<_> = map.into_iter().map(|(k, _)| k).collect();

        for x in blacklist.lock().unwrap().iter() {
            assert!(remaining.binary_search(x).is_err());
        }
    }
}

fn stress_iter(limit: u32, num_permanent: usize) {
    println!("stress_iter({}, {})", limit, num_permanent);

    let mut rng = thread_rng();
    let deadline = Instant::now() + Duration::from_millis(RUN_MILLIS);
    let map = SkipMap::new();

    let mut permanent = HashSet::new();
    while permanent.len() < num_permanent {
        permanent.insert(rng.gen_range(0, limit));
    }
    let sum_permanent: u32 = permanent.iter().sum();
    for &x in &permanent {
        map.insert(x, x);
    }

    crossbeam::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                let mut rng = thread_rng();

                while Instant::now() < deadline {
                    for _ in 0..1000 {
                        let mut x;
                        loop {
                            x = rng.gen_range(0, limit);
                            if !permanent.contains(&x) {
                                break;
                            }
                        }

                        if rng.gen() {
                            map.insert(x, 0);
                        } else {
                            map.remove(&x);
                        }
                    }
                }
            });
        }

        scope.spawn(|| {
            while Instant::now() < deadline {
                let entries: Vec<_> = map.iter().map(|e| (*e.key(), *e.value())).collect();
                for w in entries.windows(2) {
                    assert!(w[0] < w[1]);
                }

                let sum_values: u32 = entries.iter().map(|&(_, v)| v).sum();
                assert_eq!(sum_values, sum_permanent);
            }
        });

        scope.spawn(|| {
            while Instant::now() < deadline {
                let entries: Vec<_> = map.iter().rev().map(|e| (*e.key(), *e.value())).collect();
                for w in entries.windows(2) {
                    assert!(w[0] > w[1]);
                }

                let sum_values: u32 = entries.iter().map(|&(_, v)| v).sum();
                assert_eq!(sum_values, sum_permanent);
            }
        });
    });
}

fn main() {
    // TODO: random panics
    // TODO: test with broken ordering
    // TODO: count drops

    stress_small(8, 5);
    stress_small(8, 50);
    stress_small(16, 1000);
    stress_small(64, 500);

    stress_large(2);
    stress_large(8);
    stress_large(32);
    stress_large(512);

    stress_iter(5, 0);
    stress_iter(50, 0);
    stress_iter(10, 5);
    stress_iter(500, 50);
}
