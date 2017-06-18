extern crate neurons;
extern crate rand;

type Input = [f64; 2];
type Output = [f64; 2];

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
        let point = [theta.cos() + dx, theta.sin() + dy];
        let class = if point[0] * point[1] > 0.0 {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        };
        data.push((point, class));
    }
    data
}

fn score(set_name: &str,
         network: &neurons::feed_forward::Network,
         test_data: &[(Input, Output)]) {
    let mut num_correct = 0;
    for &(input, expected) in test_data {
        let output = network.run(&input);
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
    use neurons::activator::Activator;
    use neurons::feed_forward::*;

    let training_data = generate_data(10_000);
    let network = Trainer::new(&[2, 5, 5, 2])
        .activator(Activator::Sigmoid)
        .stop_condition(StopCondition::ErrorThreshold(0.001))
        .logging(Logging::Iterations(50))
        .train(&training_data)
        .unwrap();

    println!();
    score("Training", &network, &training_data);
    score("Test", &network, &generate_data(1_000));
}
