//! Utilities for training neural networks.

use std::time::{Duration, Instant};

/// Makes a model trainable using gradient desecent.
pub trait Trainable {
    /// Input data format.
    type Input;

    /// Output data format.
    type Output;

    /// A container for training updates.
    type Update;

    /// Returns a new, empty model update.
    fn new_update(&self) -> Self::Update;

    /// Using the provided training example, accumulate model updates into
    /// `update`. Returns the mean square error for the example prediction.
    fn compute_update(
        &self,
        example: &Self::Input,
        expected: &Self::Output,
        update: &mut Self::Update,
    ) -> f64;

    /// Applies and resets the provided `update`, scaling by the gradient
    /// desecent `rate`.
    fn apply_update(&mut self, rate: f64, update: &mut Self::Update);
}

/// A builder for training new models.
#[derive(Debug)]
pub struct Trainer<T: Trainable> {
    model: T,
    learning_mode: LearningMode,
    learning_rate: f64,
    logging: Logging,
    stop_condition: StopCondition,
}

impl<T: Trainable> Trainer<T> {
    /// Creates a new Trainer instance.
    ///
    /// The trainer is initialized with some default values. These defaults are:
    ///
    /// * A stochastic learning mode.
    /// * A learning rate of 0.1.
    /// * Stops after 1000 training iterations.
    /// * Logs on training completion.
    pub fn new(model: T) -> Self {
        Trainer {
            model: model,
            learning_mode: LearningMode::Stochastic,
            learning_rate: 0.1,
            logging: Logging::Completion,
            stop_condition: StopCondition::Iterations(1000),
        }
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
    pub fn stop_condition<C>(mut self, condition: C) -> Self
    where
        C: Into<StopCondition>,
    {
        self.stop_condition = condition.into();
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
    pub fn train(mut self, examples: &[(T::Input, T::Output)]) -> T {
        let mut updates = self.model.new_update();

        let start_time = Instant::now();
        let batch_size = match self.learning_mode {
            LearningMode::Stochastic => 1,
            LearningMode::Batch(size) => size,
        };
        let mut iteration = 0;
        let mut training_error;
        loop {
            training_error = 0.0;
            for (i, &(ref example, ref expected)) in examples.iter().enumerate()
            {
                training_error +=
                    self.model.compute_update(example, expected, &mut updates);
                if i % batch_size == 0 || i == examples.len() {
                    self.model.apply_update(self.learning_rate, &mut updates);
                }
            }
            training_error /= 2.0 * examples.len() as f64;
            iteration += 1;

            self.logging.iteration(iteration, training_error);
            if self.stop_condition.should_stop(
                iteration,
                training_error,
                start_time,
            ) {
                break;
            }
        }
        self.logging
            .completion(iteration, training_error, start_time);
        self.model
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
    fn completion(
        &self,
        iterations: usize,
        training_error: f64,
        start_time: Instant,
    ) {
        if let &Logging::Silent = self {
            return;
        }
        println!(
            "Ran {} iterations in {} seconds.",
            iterations,
            start_time.elapsed().as_secs()
        );
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
    /// Stops after the provided duration
    Duration(Duration),
}

impl From<Duration> for StopCondition {
    fn from(duration: Duration) -> StopCondition {
        StopCondition::Duration(duration)
    }
}

impl StopCondition {
    /// Returns true of training is complete.
    fn should_stop(
        &self,
        iteration: usize,
        training_error: f64,
        start_time: Instant,
    ) -> bool {
        use self::StopCondition::*;
        match self {
            &Iterations(iterations) => iteration >= iterations,
            &ErrorThreshold(threshold) => training_error < threshold,
            &Duration(duration) => start_time.elapsed() > duration,
        }
    }
}
