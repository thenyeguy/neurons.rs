//! Activation function types.

/// [Activation function](https://en.wikipedia.org/wiki/Activation_function)
/// types.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Activator {
    /// Rectified Linear Unit
    ReLU,
    /// Leaky Rectified Linear Unit
    ///
    /// Takes an `alpha` value to use for negative inputs.
    LeakyReLU(f64),
    /// Sigmoid function
    Sigmoid,
    /// Hyperbolic tan function
    TanH,
}

impl Activator {
    /// Evaluates `f(x)` for the selected the activation function.
    pub fn f(&self, x: f64) -> f64 {
        match self {
            &Activator::ReLU => if x > 0.0 { x } else { 0.0 },
            &Activator::LeakyReLU(alpha) => if x > 0.0 { x } else { alpha * x },
            &Activator::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            &Activator::TanH => 2.0 / (1.0 + (-2.0 * x).exp()) - 1.0,
        }
    }

    /// Evaluates the derivative `f'(x)`, where `x = f^{-1}(y)`.
    ///
    /// Note that this function takes in the *output* of the activation
    /// function, rather than the input. This is an optimization that means we
    /// don't have to store the intermediate results before activation.
    pub fn fprime(&self, y: f64) -> f64 {
        match self {
            &Activator::ReLU => if y > 0.0 { 1.0 } else { 0.0 },
            &Activator::LeakyReLU(alpha) => if y > 0.0 { 1.0 } else { alpha },
            &Activator::Sigmoid => y * (1.0 - y),
            &Activator::TanH => 1.0 - y * y,
        }
    }
}
