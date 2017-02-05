//! A [Feedforward neural network]
//! (https://en.wikipedia.org/wiki/Feedforward_neural_network).
//!
//! # Example
//!
//! Let's train a simple neural network to compute the XOR function:
//!
//! ```
//! # use neurons::feed_forward::*;
//!
//! // Create examples of the XOR function
//! let examples = [([0.0, 0.0], [0.0]),
//!                 ([0.0, 1.0], [1.0]),
//!                 ([1.0, 0.0], [1.0]),
//!                 ([1.0, 1.0], [0.0])];
//!
//! // Train a network using those examples
//! let network = Trainer::new(&[2, 3, 1])
//!     .activator(Activator::ReLU)
//!     .stop_condition(StopCondition::ErrorThreshold(0.01))
//!     .train(&examples[..])
//!     .unwrap();
//!
//! // And verify the network correctly computes XOR!
//! fn classify(out: Vec<f64>) -> bool {
//!     out[0] > 0.5
//! }
//! assert_eq!(classify(network.run(&[0.0, 0.0])), false);
//! assert_eq!(classify(network.run(&[0.0, 1.0])), true);
//! assert_eq!(classify(network.run(&[1.0, 0.0])), true);
//! assert_eq!(classify(network.run(&[1.0, 1.0])), false);
//! ```

use rblas::attribute::Transpose;
use rblas::math::mat::Mat;
use rblas::matrix_vector::ops::{Gemv, Ger};

/// [Activation function](https://en.wikipedia.org/wiki/Activation_function) types
#[derive(Copy, Clone, Debug)]
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
    fn f(&self, x: f64) -> f64 {
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
    fn fprime(&self, y: f64) -> f64 {
        match self {
            &Activator::ReLU => if y > 0.0 { 1.0 } else { 0.0 },
            &Activator::LeakyReLU(alpha) => if y > 0.0 { 1.0 } else { alpha },
            &Activator::Sigmoid => y * (1.0 - y),
            &Activator::TanH => 1.0 - y * y,
        }
    }
}

/// A wrapper for a single layer of the neural network
///
/// This performs efficient network updates by storing the weights for every
/// neuron as a single Matrix.
#[derive(Debug)]
struct Layer {
    /// The activation function to be used for every neuron in the layer.
    activator: Activator,
    /// The network weights, with each neuron's weights stored as a column.
    weights: Mat<f64>,
}

impl Layer {
    /// Initializes a new, untrained layer.
    ///
    /// Arguments:
    ///
    ///  * `activator` - the activation function to be used for this layer's
    ///                  output.
    ///  * `inputs` - the number of inputs to this layer.
    ///  * `outputs` - the number of outputs from this layer.
    fn new(activator: Activator, inputs: usize, outputs: usize) -> Self {
        Layer {
            activator: activator,
            weights: Mat::fill(1.0, outputs, inputs),
        }
    }

    /// Returns the number of inputs to this layer.
    fn input_len(&self) -> usize {
        self.weights.cols()
    }

    /// Returns the number of inputs from this layer.
    fn output_len(&self) -> usize {
        self.weights.rows()
    }

    /// Feeds the provided `inputs` forward through the layer.
    fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        assert_eq!(inputs.len(), self.input_len());
        let mut outputs = vec![0.0; self.output_len()];
        f64::gemv(Transpose::NoTrans,
                  &1.0,
                  &self.weights,
                  inputs,
                  &1.0,
                  &mut outputs);
        for y in &mut outputs {
            *y = self.activator.f(*y);
        }
        outputs
    }

    /// Feeds the provided `costs` backwards through the layer.
    ///
    /// Arguments:
    ///
    ///  * `outputs` - the outputs from this layer from the corresponding
    ///               forward pass.
    ///  * `costs` - the cost delta from the next layer.
    ///
    /// Output:
    ///   The input cost delta to this layer.
    fn backward(&self, outputs: &[f64], costs: &[f64]) -> Vec<f64> {
        assert_eq!(outputs.len(), self.output_len());
        assert_eq!(costs.len(), self.output_len());
        let mut deltas = vec![0.0; self.input_len()];
        f64::gemv(Transpose::Trans,
                  &1.0,
                  &self.weights,
                  costs,
                  &1.0,
                  &mut deltas);
        for (y, d) in outputs.iter().zip(deltas.iter_mut()) {
            *d *= self.activator.fprime(*y);
        }
        deltas
    }

    /// Updates the weights of this layer.
    ///
    /// Arguments:
    ///
    ///  * `rate` - the learning rate to be used.
    ///  * `inputs` - the input to this layer from the corresponding forward
    ///               pass.
    ///  * `deltas` - the cost delta for this layer's output.
    fn update(&mut self, rate: f64, inputs: &[f64], deltas: &[f64]) {
        assert_eq!(inputs.len(), self.input_len());
        assert_eq!(deltas.len(), self.output_len());
        f64::ger(&(-rate), inputs, deltas, &mut self.weights);
    }
}

/// A Feedforward neural network
#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    /// Creates a new, untrained neural network.
    ///
    /// Arguments:
    ///  * `activator` - the activation function to use for each neuron.
    ///  * `layer_sizes` - the number of neurons in each layer.
    fn new(activator: Activator, layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(Layer::new(activator,
                                   layer_sizes[i],
                                   layer_sizes[i + 1]));
        }
        Network { layers: layers }
    }

    /// Returns the size of the input layer to the network.
    pub fn input_len(&self) -> usize {
        self.layers.front().input_len()
    }

    /// Returns the size of the output layer from the network.
    pub fn output_len(&self) -> usize {
        self.layers.back().output_len()
    }

    /// Feeds the provided `input` through the network, returning the output
    /// layer.
    pub fn run(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.layers.front().input_len());
        self.feed_forward(input).pop().expect("missing layer outputs")
    }

    /// Feeds the provided `input` through the network, returning the activated
    /// values for each layer.
    fn feed_forward(&self, input: &[f64]) -> Vec<Vec<f64>> {
        let mut network = Vec::with_capacity(self.layers.len());
        network.push(input.to_vec());
        for layer in &self.layers {
            let output = layer.forward(network.back());
            network.push(output);
        }
        network
    }

    /// Feeds the provided `expected` value back through the network, returning
    /// the computed cost deltas.
    ///
    /// Arguments:
    ///
    ///  * `network` - the activated values for each layer in the network. This
    ///                should be the output of `self.feed_forward(...)`.
    ///  * `expected` - the expected output of the network given the input.
    fn feed_backwards(&self,
                      network: &[Vec<f64>],
                      expected: &[f64])
        -> Vec<Vec<f64>> {
            let mut deltas = Vec::with_capacity(self.layers.len());

            let mut output_delta = Vec::with_capacity(expected.len());
            for (o, e) in network.back().iter().zip(expected.iter()) {
                output_delta.push(o - e);
            }
            deltas.push(output_delta);

            for i in (0..self.layers.len()).rev() {
                let delta = self.layers[i]
                    .backward(&network[i + 1], &deltas.back());
                deltas.push(delta);
            }
            deltas.reverse();
            deltas
        }

    /// Updates the network weights.
    ///
    /// Arguments:
    ///
    ///  * `rate` - the learning to use for gradient descent.
    ///  * `network` - the activated values of each layer in the network. This
    ///                should be the output of `self.feed_forward(...)`.
    ///  * `deltas` - the cost deltas, computed from `self.feed_backwards(...)`.
    fn update(&mut self,
              rate: f64,
              network: &[Vec<f64>],
              deltas: &[Vec<f64>]) {
        for (layer, (delta, input)) in
            self.layers.iter_mut().zip(deltas[1..].iter().zip(network.iter())) {
                layer.update(rate, input, delta);
            }
    }
}

/// The learning mode to use for training
#[derive(Copy, Clone, Debug)]
pub enum LearningMode {
    /// Apply weight updates after every training example
    Stochastic,
}

/// Logging frequency to use during training
#[derive(Copy, Clone, Debug)]
pub enum Logging {
    /// No logs will be printed
    Silent,
    /// A summary will be printed at completion
    Completion,
    /// A summary will be printed after every `n` training iterations
    Iterations(usize),
}

impl Logging {
    /// Performs logging at the current `iteration` of training.
    fn iteration(&self, iteration: usize, training_error: f64) {
        use self::Logging::*;
        if let &Iterations(freq) = self {
            if freq > 0 && iteration % freq == 0 {
                println!("Iteration {}:\tMSE={}", iteration, training_error);
            }
        }
    }

    /// Performs logging at the end of training.
    fn completion(&self, iterations: usize, training_error: f64) {
        if let &Logging::Silent = self {
            return;
        }
        println!("Training completed after {} iterations.", iterations);
        println!("Final MSE: {}", training_error);
    }
}

/// When to stop training
#[derive(Copy, Clone, Debug)]
pub enum StopCondition {
    /// Stops after the provided number of training iterations
    Iterations(usize),
    /// Stops when the training error drops below the provided threshold
    ErrorThreshold(f64),
}

impl StopCondition {
    /// Returns true of training is complete.
    fn should_stop(&self, iteration: usize, training_error: f64) -> bool {
        use self::StopCondition::*;
        match self {
            &Iterations(iterations) => iteration >= iterations,
            &ErrorThreshold(threshold) => training_error < threshold,
        }
    }
}

/// Trains a new `Network` object
#[derive(Debug)]
pub struct Trainer {
    layer_sizes: Vec<usize>,
    activator: Activator,
    learning_mode: LearningMode,
    learning_rate: f64,
    logging: Logging,
    stop_condition: StopCondition,
}

impl Trainer {
    /// Creates a new Trainer instance.
    ///
    /// Arguments:
    ///  * `layers` - the number of neurons to use at each layer. Must contain
    ///               at least 3 elements - one input layer, one hidden layer,
    ///               and one output layer.
    ///
    /// The trainer is initialized with some default values. These defaults are:
    ///
    /// * A ReLU activation function.
    /// * A stochastic learning mode.
    /// * A learning rate of 0.1.
    /// * Stops after 1000 training iterations.
    /// * Logs on training completion.
    pub fn new(layers: &[usize]) -> Self {
        Trainer {
            layer_sizes: layers.into(),
            activator: Activator::ReLU,
            learning_mode: LearningMode::Stochastic,
            learning_rate: 0.1,
            logging: Logging::Completion,
            stop_condition: StopCondition::Iterations(1000),
        }
    }

    /// Sets the activation function to use in the network.
    pub fn activator(mut self, activator: Activator) -> Self {
        self.activator = activator;
        self
    }

    /// Sets the `LearningMode` to use for training.
    pub fn learning_mode(mut self, mode: LearningMode) -> Self {
        self.learning_mode = mode;
        self
    }

    /// Sets the learning rate to use during gradient descent.
    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Sets the type of logging to be emitted during training.
    pub fn logging(mut self, logging: Logging) -> Self {
        self.logging = logging;
        self
    }

    /// Sets the condition to finish training.
    pub fn stop_condition(mut self, condition: StopCondition) -> Self {
        self.stop_condition = condition;
        self
    }

    /// Trains a network using the provided labelled data.
    ///
    /// The provided `examples` should be a list of labelled data, where each
    /// element takes the form `(network input, expected output)`.
    ///
    /// Returns:
    ///   A fully trained neural network, or an error if invalid training
    ///   parameters were provided.
    pub fn train<I, O>(self, examples: &[(I, O)]) -> Result<Network, ()>
        where I: AsRef<[f64]>,
              O: AsRef<[f64]>
              {
                  try!(self.validate(examples));

                  let mut network = Network::new(self.activator, &self.layer_sizes);
                  let mut iteration = 0;
                  let mut training_error;
                  loop {
                      training_error = 0.0;
                      for &(ref input, ref expected) in examples {
                          let activations = network.feed_forward(input.as_ref());
                          let deltas =
                              network.feed_backwards(&activations, expected.as_ref());
                          network.update(self.learning_rate, &activations, &deltas);
                          training_error += mean_square_error(activations.back(),
                          expected.as_ref());
                      }
                      training_error /= 2.0 * examples.len() as f64;
                      iteration += 1;
                      self.logging.iteration(iteration, training_error);
                      if self.stop_condition
                          .should_stop(iteration, training_error) {
                              break;
                          }
                  }
                  self.logging.completion(iteration, training_error);
                  Ok(network)
              }

    /// Verifies that all provided inputs to the `Trainer` are valid, returning
    /// an error if something is wrong.
    fn validate<I, O>(&self, examples: &[(I, O)]) -> Result<(), ()>
        where I: AsRef<[f64]>,
              O: AsRef<[f64]>
              {
                  if self.layer_sizes.len() < 3 {
                      return Err(());
                  }
                  for layer_size in &self.layer_sizes {
                      if *layer_size == 0 {
                          return Err(());
                      }
                  }
                  for &(ref input, ref output) in examples {
                      if input.as_ref().len() != *self.layer_sizes.front() {
                          return Err(());
                      }
                      if output.as_ref().len() != *self.layer_sizes.back() {
                          return Err(());
                      }
                  }
                  Ok(())
              }
}

/// Computes the mean squared error between `actual` and `expected`.
fn mean_square_error(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut error = 0.0;
    for (&a, e) in actual.iter().zip(expected) {
        error += (a - e) * (a - e);
    }
    error / (actual.len() as f64)
}

/// A trait that provides easy access to slice elements.
trait Items<T> {
    #[inline(always)]
    fn front(&self) -> &T;
    #[inline(always)]
    fn mut_front(&mut self) -> &mut T;
    #[inline(always)]
    fn back(&self) -> &T;
    #[inline(always)]
    fn mut_back(&mut self) -> &mut T;
}

impl<T> Items<T> for [T] {
    #[inline(always)]
    fn front(&self) -> &T {
        &self[0]
    }
    #[inline(always)]
    fn mut_front(&mut self) -> &mut T {
        &mut self[0]
    }
    #[inline(always)]
    fn back(&self) -> &T {
        &self[self.len() - 1]
    }
    #[inline(always)]
    fn mut_back(&mut self) -> &mut T {
        let i = self.len() - 1;
        &mut self[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_layers() {
        let examples = [([0.0], [0.0])];
        assert!(Trainer::new(&[1]).train(&examples[..]).is_err());
    }

    #[test]
    fn empty_layer() {
        let examples = [([0.0], [0.0])];
        assert!(Trainer::new(&[1, 0, 1]).train(&examples[..]).is_err());
    }

    #[test]
    fn wrong_input_size() {
        let examples = [([0.0, 0.0], [0.0])];
        assert!(Trainer::new(&[1, 1]).train(&examples[..]).is_err());
    }

    #[test]
    fn wrong_output_size() {
        let examples = [([0.0], [0.0, 0.0])];
        assert!(Trainer::new(&[1, 1]).train(&examples[..]).is_err());
    }
}
