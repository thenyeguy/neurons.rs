//! A [Feedforward neural network]
//! (https://en.wikipedia.org/wiki/Feedforward_neural_network).
//!
//! # Example
//!
//! Let's train a simple neural network to compute the XOR function:
//!
//! ```
//! # use neurons::feed_forward::*;
//! # use neurons::trainer::*;
//! // Create examples of the XOR function
//! let examples = [(vec![0.0, 0.0], vec![0.0]),
//!                 (vec![0.0, 1.0], vec![1.0]),
//!                 (vec![1.0, 0.0], vec![1.0]),
//!                 (vec![1.0, 1.0], vec![0.0])];
//!
//! // Train a network using those examples
//! let model = Trainer::new(Model::new(Activator::Sigmoid, &[2, 3, 1]))
//!     .learning_rate(0.5)
//!     .stop_condition(StopCondition::Iterations(30000))
//!     .train(&examples[..]);
//!
//! // And verify the network correctly computes XOR!
//! let mut runner = Runner::new(&model);
//! fn classify(out: &[f64]) -> bool {
//!     out[0] > 0.5
//! }
//! assert_eq!(classify(runner.run(&[0.0, 0.0])), false);
//! assert_eq!(classify(runner.run(&[0.0, 1.0])), true);
//! assert_eq!(classify(runner.run(&[1.0, 0.0])), true);
//! assert_eq!(classify(runner.run(&[1.0, 1.0])), false);
//! ```

use layers::{dense, Layer};
use trainer::Trainable;
use utils::{Front, Back, ZeroOut};

use itertools::zip;

pub use activator::Activator;

/// A feed-forward neural network model.
#[derive(Debug)]
pub struct Model {
    layers: Vec<dense::Layer>,
}

impl Model {
    /// Creates a new, untrained model.
    ///
    /// Arguments:
    ///
    ///  * `activator` - the activation function to use for each neuron.
    ///  * `layer_sizes` - the number of neurons in each layer.
    pub fn new(activator: Activator, layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..(layer_sizes.len() - 1) {
            layers.push(dense::Layer::new(activator,
                                          layer_sizes[i],
                                          layer_sizes[i + 1]));
        }
        Model { layers: layers }
    }

    /// Returns the size of the input layer to the network.
    pub fn input_len(&self) -> usize {
        self.layers.front().input_len()
    }

    /// Returns the size of the output layer from the network.
    pub fn output_len(&self) -> usize {
        self.layers.back().output_len()
    }

    /// Feeds the provided `input` through the network, returning the activated
    /// values for each layer.
    fn feed_forward(&self, input: &[f64], network: &mut [Vec<f64>]) {
        network[0].copy_from_slice(input);
        for (i, layer) in self.layers.iter().enumerate() {
            let (input, output) = mut_layers(network, i);
            layer.forward(input, output, &mut ());
        }
    }

    /// Feeds the provided `expected` value back through the network, returning
    /// the computed cost deltas.
    fn feed_backwards(&self,
                      network: &[Vec<f64>],
                      expected: &[f64],
                      errors: &mut [Vec<f64>],
                      updates: &mut [dense::Update]) {
        for i in 0..expected.len() {
            errors.mut_back()[i] = expected[i] - network.back()[i];
        }
        for (i, layer) in (self.layers.iter().enumerate()).rev() {
            let (inputs, outputs) = io_layers(network, i);
            let (in_error, out_error) = mut_layers(errors, i);
            layer.backward(inputs,
                           outputs,
                           &(),
                           out_error,
                           in_error,
                           &mut updates[i]);
        }
    }

    // Returns a fully zeroed activation network.
    fn empty_activation_network(&self) -> Vec<Vec<f64>> {
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(vec![0.0; self.layers.front().input_len()]);
        for layer in &self.layers {
            activations.push(vec![0.0; layer.output_len()]);
        }
        activations
    }
}

/// A container for training updates.
pub struct ModelUpdate {
    // These fields simply hold buffers for running backpropogation.
    activations: Vec<Vec<f64>>,
    errors: Vec<Vec<f64>>,

    // This field holds the actual layer updates.
    updates: Vec<dense::Update>,
}

impl Trainable for Model {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type Update = ModelUpdate;

    fn new_update(&self) -> Self::Update {
        let mut layer_updates = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layer_updates.push(layer.new_update());
        }

        ModelUpdate {
            activations: self.empty_activation_network(),
            errors: self.empty_activation_network(),
            updates: layer_updates,
        }
    }

    fn compute_update(&self,
                      example: &Self::Input,
                      expected: &Self::Output,
                      update: &mut Self::Update)
                      -> f64 {
        update.activations.zero_out();
        update.errors.zero_out();
        self.feed_forward(&example, &mut update.activations);
        self.feed_backwards(&update.activations,
                            &expected,
                            &mut update.errors,
                            &mut update.updates);
        mean_square_error(update.activations.back(), &expected)
    }

    fn apply_update(&mut self, rate: f64, update: &mut Self::Update) {
        for (layer, weight_update) in
            zip(&mut self.layers, &mut update.updates) {
            layer.apply_update(rate, weight_update);
        }
    }
}

/// Runs trained network models.
#[derive(Debug)]
pub struct Runner<'a> {
    model: &'a Model,
    activations: Vec<Vec<f64>>,
}

impl<'a> Runner<'a> {
    /// Creates a new Runner using the provided model.
    pub fn new(model: &'a Model) -> Self {
        Runner {
            model: model,
            activations: model.empty_activation_network(),
        }
    }

    /// Feeds the provided `input` through the network, and returns the output
    /// layer.
    pub fn run(&mut self, input: &[f64]) -> &[f64] {
        self.activations.zero_out();
        self.model.feed_forward(input, &mut self.activations);
        &self.activations.back()
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
    for (&a, &e) in zip(actual, expected) {
        error += (a - e) * (a - e);
    }
    error / (actual.len() as f64)
}
