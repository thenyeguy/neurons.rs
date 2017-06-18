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

use activator::Activator;
use layer::Layer;
use matrix::Mat;
use traits::{Front, Back, ZeroOut};

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
            let (input, output) = mut_layers(network, i);
            layer.forward(input, output);
        }
    }

    /// Feeds the provided `expected` value back through the network, returning
    /// the computed cost deltas.
    fn feed_backwards(&self,
                      learning_rate: f64,
                      network: &[Vec<f64>],
                      expected: &[f64],
                      errors: &mut [Vec<f64>],
                      weight_updates: &mut [Mat]) {
        for i in 0..expected.len() {
            errors.mut_back()[i] = expected[i] - network.back()[i];
        }
        for (i, layer) in (self.layers.iter().enumerate()).rev() {
            let (inputs, outputs) = io_layers(network, i);
            let (in_error, out_error) = mut_layers(errors, i);
            layer.backward(learning_rate,
                           inputs,
                           outputs,
                           in_error,
                           out_error,
                           &mut weight_updates[i]);
        }
    }

    /// Applies weight updates to the network.
    fn update(&mut self, weight_updates: &[Mat]) {
        for (layer, weight_update) in
            self.layers.iter_mut().zip(weight_updates.iter()) {
            layer.apply_update(weight_update);
        }
    }

    /// Returns an activation network full of zeros.
    fn empty_network(&self) -> Vec<Vec<f64>> {
        let mut network = Vec::with_capacity(self.layers.len() + 1);
        network.push(vec![0.0; self.layers.front().input_len()]);
        for layer in &self.layers {
            network.push(vec![0.0; layer.output_len()]);
        }
        network
    }

    /// Returns a zeroed vector of weights updates for each layer.
    fn empty_weight_updates(&self) -> Vec<Mat> {
        let mut updates = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            updates.push(layer.empty_weight_update());
        }
        updates
    }
}

/// The learning mode to use for training
#[derive(Copy, Clone, Debug)]
pub enum LearningMode {
    /// Apply weight updates after every training example
    Stochastic,
    /// Apply weights updates in batches of the provided size
    ///
    /// Must be less than the total number of training instances.
    Batch(usize),
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
        let mut updates = network.empty_weight_updates();

        let batch_size = match self.learning_mode {
            LearningMode::Stochastic => 1,
            LearningMode::Batch(size) => size,
        };
        let mut iteration = 0;
        let mut training_error;
        loop {
            training_error = 0.0;
            for (i, &(ref input, ref expected)) in examples.iter().enumerate() {
                activations.zero_out();
                errors.zero_out();
                network.feed_forward(input.as_ref(), &mut activations);
                network.feed_backwards(self.learning_rate,
                                       &activations,
                                       expected.as_ref(),
                                       &mut errors,
                                       &mut updates);
                if i % batch_size == 0 || i == examples.len() {
                    network.update(&updates);
                    updates.zero_out();
                }
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
        if let LearningMode::Batch(batch_size) = self.learning_mode {
            if batch_size > examples.len() {
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
fn io_layers(layers: &[Vec<f64>], layer: usize) -> (&[f64], &[f64]) {
    let (before, after) = layers[layer..].split_at(1);
    (&before[0], &after[0])
}

fn mut_layers(layers: &mut [Vec<f64>],
              layer: usize)
              -> (&mut [f64], &mut [f64]) {
    let (before, after) = layers[layer..].split_at_mut(1);
    (&mut before[0], &mut after[0])
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

    #[test]
    fn too_large_batch_size() {
        let examples = [([0.0], [0.0])];
        assert!(Trainer::new(&[1])
            .learning_mode(LearningMode::Batch(2))
            .train(&examples[..])
            .is_err());
    }
}
