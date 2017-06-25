extern crate itertools;
extern crate rand;
extern crate rblas;
#[macro_use]
extern crate serde_derive;

pub mod activator;
pub mod feed_forward;
pub mod trainer;

mod layers;
mod matrix;
mod utils;
