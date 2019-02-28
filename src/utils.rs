/// A trait that provides easy access to the first element of a slice.
pub trait Front<T> {
    fn front(&self) -> &T;
    fn mut_front(&mut self) -> &mut T;
}

/// A trait that provides easy access to the last element of a slice.
pub trait Back<T> {
    fn back(&self) -> &T;
    fn mut_back(&mut self) -> &mut T;
}

impl<T> Front<T> for [T] {
    fn front(&self) -> &T {
        &self[0]
    }
    fn mut_front(&mut self) -> &mut T {
        &mut self[0]
    }
}

impl<T> Back<T> for [T] {
    fn back(&self) -> &T {
        &self[self.len() - 1]
    }
    fn mut_back(&mut self) -> &mut T {
        let i = self.len() - 1;
        &mut self[i]
    }
}

/// A trait to replace all elements in a container with zeros.
pub trait ZeroOut {
    fn zero_out(&mut self);
}

impl ZeroOut for f64 {
    fn zero_out(&mut self) {
        *self = 0.0;
    }
}

impl<T> ZeroOut for [T]
where
    T: ZeroOut,
{
    fn zero_out(&mut self) {
        for elem in self {
            elem.zero_out();
        }
    }
}

impl<T> ZeroOut for Vec<T>
where
    T: ZeroOut,
{
    fn zero_out(&mut self) {
        for elem in self {
            elem.zero_out();
        }
    }
}
