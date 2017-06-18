use activator::Activator;
use matrix::Mat;

use rand::distributions::Normal;
use rblas::Matrix;
use rblas::attribute::Transpose;
use rblas::matrix_vector::ops::{Gemv, Ger};

/// A wrapper for a single layer of the neural network
///
/// This performs efficient network updates by storing the weights for every
/// neuron as a single Matrix.
#[derive(Debug)]
pub struct Layer {
    /// The activation function to be used for every neuron in the layer.
    activator: Activator,
    /// The network weights, with each neuron's weights stored as a column.
    weights: Mat,
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
    pub fn new(activator: Activator, inputs: usize, outputs: usize) -> Self {
        Layer {
            activator: activator,
            weights: Mat::random(Normal::new(0.0, 1.0), outputs, inputs),
        }
    }

    /// Returns the number of inputs to this layer.
    pub fn input_len(&self) -> usize {
        self.weights.cols() as usize
    }

    /// Returns the number of inputs from this layer.
    pub fn output_len(&self) -> usize {
        self.weights.rows() as usize
    }

    /// Feeds the provided `inputs` forward through the layer.
    pub fn forward(&self, inputs: &[f64], outputs: &mut [f64]) {
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
    pub fn backward(&self,
                learning_rate: f64,
                inputs: &[f64],
                outputs: &[f64],
                input_errors: &mut [f64],
                output_errors: &mut [f64],
                weight_updates: &mut Mat) {
        assert_eq!(inputs.len(), self.input_len());
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
        f64::ger(&learning_rate, output_errors, inputs, weight_updates);
    }

    pub fn apply_update(&mut self, update: &Mat) {
        self.weights += update;
    }

    /// Returns an empty weight update matrix.
    pub fn empty_weight_update(&self) -> Mat {
        Mat::zeros(self.output_len(), self.input_len())
    }
}

