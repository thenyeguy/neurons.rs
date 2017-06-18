extern crate neurons;
extern crate rand;

type Input = Vec<f64>;
type Output = Vec<f64>;

fn generate_data(num_samples: usize) -> Vec<(Input, Output)> {
    use rand::distributions::IndependentSample;
    let mut rng = rand::thread_rng();
    let radians = rand::distributions::Range::new(0.0,
                                                  2.0 * std::f64::consts::PI);
    let noise = rand::distributions::Normal::new(0.0, 0.1);

    let mut data = Vec::new();
    for _ in 0..num_samples {
        let theta = radians.ind_sample(&mut rng);
        let dx = noise.ind_sample(&mut rng);
        let dy = noise.ind_sample(&mut rng);
        let point = vec![theta.cos() + dx, theta.sin() + dy];
        let class = if point[0] * point[1] > 0.0 {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        };
        data.push((point, class));
    }
    data
}

fn score(set_name: &str,
         test_data: &[(Input, Output)],
         runner: &mut neurons::feed_forward::Runner) {
    let mut num_correct = 0;
    for &(ref input, ref expected) in test_data {
        let output = runner.run(&input);
        let class = if output[0] > output[1] { 0 } else { 1 };
        if expected[class] == 1.0 {
            num_correct += 1;
        }
    }
    println!("{} set results: {} of {} correct",
             set_name,
             num_correct,
             test_data.len());
}

fn main() {
    use neurons::feed_forward::{self, Activator};
    use neurons::trainer::*;

    let training_data = generate_data(10_000);
    let model = Trainer::new(feed_forward::Model::new(Activator::Sigmoid,
                                                      &[2, 5, 5, 2]))
        .stop_condition(StopCondition::ErrorThreshold(0.001))
        .logging(Logging::Iterations(50))
        .train(&training_data);
    let mut runner = feed_forward::Runner::new(&model);

    println!();
    score("Training", &training_data, &mut runner);
    score("Test", &generate_data(1_000), &mut runner);
}
