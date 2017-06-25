pub mod dense;

pub trait Layer {
    type Update;

    /// The size of inputs to this layer.
    fn input_len(&self) -> usize;

    /// The size of outputs from this layer.
    fn output_len(&self) -> usize;

    /// Returns a new, empty model update.
    fn new_update(&self) -> Self::Update;

    /// Feeds the provided inputs forward through the layer.
    fn forward(&self, inputs: &[f64], outputs: &mut [f64]);

    /// Backpropogates errors back through the layer, and accumulates layer
    /// updates into `update`.
    fn backward(&self,
                inputs: &[f64],
                outputs: &[f64],
                output_errors: &[f64],
                input_errors: &mut [f64],
                updates: &mut Self::Update);

    /// Applies and resets the provided `update`, scaling by the gradient
    /// desecent `rate`.
    fn apply_update(&mut self, rate: f64, update: &mut Self::Update);
}
