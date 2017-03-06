//! A [Feedforward neural network]
//! (https://en.wikipedia.org/wiki/Feedforward_neural_network).
//!
//! # Example
//!
//! Let's train a simple neural network to compute the XOR function:
//!
//! ```
//! # use neurons::feed_forward::*;
//! // Create examples of the XOR function
//! let examples = [([0.0, 0.0], [0.0]),
//!                 ([0.0, 1.0], [1.0]),
//!                 ([1.0, 0.0], [1.0]),
//!                 ([1.0, 1.0], [0.0])];
//!
//! // Train a network using those examples
//! let network = Trainer::new(&[2, 3, 1])
//!     .activator(Activator::Sigmoid)
//!     .learning_rate(0.3)
//!     .logging(Logging::Iterations(1000))
//!     .stop_condition(StopCondition::Iterations(20000))
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

use traits::{Front, Back, ZeroOut};

use rand::distributions::Normal;
use rblas::Matrix;
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
            weights: rand_mat(outputs, inputs),
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
    fn forward(&self, inputs: &[f64], outputs: &mut [f64]) {
        assert_eq!(inputs.len(), self.input_len());
        assert_eq!(outputs.len(), self.output_len());
        f64::gemv(Transpose::NoTrans,
                  &1.0,
                  &self.weights,
                  inputs,
                  &1.0,
                  outputs);
        for y in outputs {
            *y = self.activator.f(*y);
        }
    }

    /// Feeds the provided `costs` backwards through the layer.
    fn backward(&self,
                outputs: &[f64],
                output_errors: &mut [f64],
                input_errors: &mut [f64]) {
        assert_eq!(outputs.len(), self.output_len());
        assert_eq!(output_errors.len(), self.output_len());
        assert_eq!(input_errors.len(), self.input_len());
        for (y, e) in outputs.iter().zip(output_errors.iter_mut()) {
            *e *= self.activator.fprime(*y);
        }
        f64::gemv(Transpose::Trans,
                  &1.0,
                  &self.weights,
                  output_errors,
                  &1.0,
                  input_errors);
    }

    /// Updates the weights of this layer.
    fn update(&mut self, rate: f64, inputs: &[f64], output_errors: &[f64]) {
        assert_eq!(inputs.len(), self.input_len());
        assert_eq!(output_errors.len(), self.output_len());
        f64::ger(&rate, output_errors, inputs, &mut self.weights);
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
        let mut network = self.empty_network();
        self.feed_forward(input, &mut network);
        network.pop().unwrap()
    }

    /// Feeds the provided `input` through the network, returning the activated
    /// values for each layer.
    fn feed_forward(&self, input: &[f64], network: &mut [Vec<f64>]) {
        network[0].copy_from_slice(input);
        for (i, layer) in self.layers.iter().enumerate() {
            let (input, output) = io_layers(network, i);
            layer.forward(input, output);
        }
    }

    /// Feeds the provided `expected` value back through the network, returning
    /// the computed cost deltas.
    fn feed_backwards(&self,
                      network: &[Vec<f64>],
                      expected: &[f64],
                      errors: &mut [Vec<f64>]) {
        for i in 0..expected.len() {
            errors.mut_back()[i] = expected[i] - network.back()[i];
        }
        for (i, layer) in (self.layers.iter().enumerate()).rev() {
            let (in_error, out_error) = io_layers(errors, i);
            layer.backward(&network[i + 1], out_error, in_error);
        }
    }

    /// Updates the network weights.
    fn update(&mut self,
              rate: f64,
              network: &[Vec<f64>],
              errors: &[Vec<f64>]) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.update(rate, &network[i], &errors[i + 1]);
        }
    }

    fn empty_network(&self) -> Vec<Vec<f64>> {
        let mut network = Vec::new();
        network.push(vec![0.0; self.layers.front().input_len()]);
        for layer in &self.layers {
            network.push(vec![0.0; layer.output_len()]);
        }
        network
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
        let mut activations = network.empty_network();
        let mut errors = network.empty_network();
        let mut iteration = 0;
        let mut training_error;
        loop {
            training_error = 0.0;
            for &(ref input, ref expected) in examples {
                activations.zero_out();
                errors.zero_out();
                network.feed_forward(input.as_ref(), &mut activations);
                network.feed_backwards(&activations,
                                       expected.as_ref(),
                                       &mut errors);
                network.update(self.learning_rate, &activations, &errors);
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

/// Gets input and output slices for a layer.
fn io_layers(layers: &mut [Vec<f64>],
             layer: usize)
             -> (&mut [f64], &mut [f64]) {
    let (before, after) = layers[layer..].split_at_mut(1);
    (&mut before[0], &mut after[0])
}

/// Generates a randomly-initialized matrix.
fn rand_mat(rows: usize, cols: usize) -> Mat<f64> {
    use rand;
    use rand::distributions::IndependentSample;
    let mut rng = rand::thread_rng();
    let range = rand::distributions::Normal::new(0.0, 1.0);

    let mut mat = Mat::new(0, 0);
    unsafe {
        for _ in 0..(rows*cols) {
            mat.push(range.ind_sample(&mut rng));
        }
        mat.set_cols(cols);
        mat.set_rows(rows);
    }
    mat
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
