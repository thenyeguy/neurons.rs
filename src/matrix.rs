use traits::ZeroOut;

use rand;
use rand::distributions::IndependentSample;
use rblas::attribute::Order;
use rblas::Matrix;
use std::ops::AddAssign;
use std::os::raw::c_int;

#[derive(Clone, Debug)]
pub struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>, // column-major array
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows: rows,
            cols: cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random<D>(distribution: D, rows: usize, cols: usize) -> Self
        where D: IndependentSample<f64>
    {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            data.push(distribution.ind_sample(&mut rng));
        }
        Mat {
            rows: rows,
            cols: cols,
            data: data,
        }
    }
}

impl<'a> AddAssign<&'a Mat> for Mat {
    fn add_assign(&mut self, other: &Mat) {
        for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
            *l += *r;
        }
    }
}

impl Matrix<f64> for Mat {
    fn rows(&self) -> c_int {
        self.rows as c_int
    }

    fn cols(&self) -> c_int {
        self.cols as c_int
    }

    fn as_ptr(&self) -> *const f64 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut f64 {
        self.data.as_mut_ptr()
    }

    fn order(&self) -> Order {
        Order::ColMajor
    }
}

impl ZeroOut for Mat {
    fn zero_out(&mut self) {
        self.data.zero_out();
    }
}
