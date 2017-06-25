trait Layer {
    type Update;

    /// Returns a new, empty model update.
    fn new_update(&self) -> Self::Update;

    fn forward(&self, inputs: &[f64], outputs: &mut [f64]);

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
