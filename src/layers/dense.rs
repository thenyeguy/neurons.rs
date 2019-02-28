use crate::activator::Activator;
use crate::layers;
use crate::matrix::Mat;
use crate::utils::ZeroOut;

use itertools::multizip;
use rand::distributions::Normal;
use rblas::attribute::Transpose;
use rblas::matrix_vector::ops::{Gemv, Ger};
use rblas::Matrix;

/// A wrapper for a fully connected layer of a neural network
///
/// This performs efficient network updates by storing the weights for every
/// neuron as a single Matrix.
#[derive(Debug, Serialize, Deserialize)]
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
            activator,
            weights: Mat::random(Normal::new(0.0, 1.0), outputs, inputs),
        }
    }
}

#[derive(Debug)]
pub struct Update {
    weight_delta: Mat,
    derivative: Vec<f64>,
}

impl layers::Layer for Layer {
    type State = ();
    type Update = Update;

    fn input_len(&self) -> usize {
        self.weights.cols() as usize
    }

    fn output_len(&self) -> usize {
        self.weights.rows() as usize
    }

    fn new_state(&self) -> Self::State {}

    fn new_update(&self) -> Self::Update {
        Update {
            weight_delta: Mat::zeros(self.output_len(), self.input_len()),
            derivative: vec![0.0; self.output_len()],
        }
    }

    fn forward(
        &self,
        inputs: &[f64],
        outputs: &mut [f64],
        _: &mut Self::State,
    ) {
        assert_eq!(inputs.len(), self.input_len());
        assert_eq!(outputs.len(), self.output_len());
        f64::gemv(
            Transpose::NoTrans,
            &1.0,
            &self.weights,
            inputs,
            &1.0,
            outputs,
        );
        for y in outputs {
            *y = self.activator.f(*y);
        }
    }

    /// Feeds the provided `costs` backwards through the layer.
    fn backward(
        &self,
        inputs: &[f64],
        outputs: &[f64],
        _: &Self::State,
        output_errors: &[f64],
        input_errors: &mut [f64],
        update: &mut Self::Update,
    ) {
        assert_eq!(inputs.len(), self.input_len());
        assert_eq!(outputs.len(), self.output_len());
        assert_eq!(output_errors.len(), self.output_len());
        assert_eq!(input_errors.len(), self.input_len());
        for (y, e, d) in multizip((
            outputs.iter(),
            output_errors.iter(),
            update.derivative.iter_mut(),
        )) {
            *d = e * self.activator.fprime(*y);
        }
        f64::gemv(
            Transpose::Trans,
            &1.0,
            &self.weights,
            &update.derivative,
            &1.0,
            input_errors,
        );
        f64::ger(&1.0, &update.derivative, inputs, &mut update.weight_delta);
    }

    fn apply_update(&mut self, rate: f64, update: &mut Self::Update) {
        self.weights.apply_delta(rate, &update.weight_delta);
        update.weight_delta.zero_out();
    }
}
