extern crate skiplist;

use std::sync::Arc;
use std::thread;

use skiplist::SkipMap;

fn main() {
    let s = SkipMap::new();
    let my = Arc::new(s);

    use std::time::{Duration, Instant};
    let now = Instant::now();

    const T: usize = 1;
    let mut v = (0..T)
        .map(|mut t| {
            let my = my.clone();
            thread::spawn(move || {
                let mut num = t as u32;
                for i in 0..10_000_000 / T {
                    num = num.wrapping_mul(17).wrapping_add(255);
                    my.insert(num, !num);
                }
            })
        })
        .collect::<Vec<_>>();
    for h in v.drain(..) {
        h.join().unwrap();
    }
    // v.extend((0..T).map(|mut t| {
    //     let my = my.clone();
    //     thread::spawn(move || {
    //         let mut num = t as u32;
    //         for i in 0..1_000_000 / T {
    //             num = num.wrapping_mul(17).wrapping_add(255);
    //             my.remove(&num);
    //         }
    //     })
    // }));
    // for h in v {
    //     h.join().unwrap();
    // }

    let elapsed = now.elapsed();

    // let elapsed = now.elapsed();
    println!(
        "seconds: {:.3}",
        elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1e9
    );
    // println!("LEN: {}", my.count());

    let now = Instant::now();
    println!("cnt = {}", my.count());
    let elapsed = now.elapsed();
    println!(
        "iteration seconds: {:.3}",
        elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1e9
    );

    my.insert(10, 0);
    my.insert(20, 0);
    my.insert(30, 0);

    // let mut c = my.cursor();
    // println!("{:?}", c.seek(&33));
    // println!("-> {:?}", c.key());
}
