use std::collections::BTreeMap;

use ball_tree::BallTree;
use itertools::Itertools;
use ndarray::prelude::*;
use polars::prelude::*;

// ADD OPTION TO SWITCH TO BRUTE FORCE

#[derive(Debug, Clone)]
struct NewType(Vec<f64>);

impl PartialEq for NewType {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            false
        } else {
            self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
        }
    }
}

impl ball_tree::Point for NewType {
    fn distance(&self, other: &Self) -> f64 {
        assert_eq!(self.0.len(), other.0.len());
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {
        let distance = self.distance(other);

        if distance == 0.0 {
            return self.clone();
        }

        let scale = d / distance;

        NewType(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| scale * (b - a))
                .collect_vec(),
        )
    }
}

#[derive(Debug)]
pub struct KNNClassifier<T> {
    data: Option<BallTree<NewType, T>>,
    k: usize,
}

#[derive(Debug, Default)]
pub struct MeasurePerformance {
    pub accuracy: f64,
}

impl MeasurePerformance {
    fn new<T: PartialEq + Eq>(pred: &Vec<T>, actual: &Vec<T>) -> Option<Self> {
        if pred.len() != actual.len() {
            None
        } else {
            let acc = pred
                .iter()
                .zip(actual.iter())
                .map(|(a, b)| if a == b { 1 } else { 0 })
                .sum::<usize>();
            Some(Self {
                accuracy: acc as f64 / pred.len() as f64,
            })
        }
    }
}

impl<T> Default for KNNClassifier<T> {
    fn default() -> Self {
        Self { k: 1, data: None }
    }
}

impl<T> KNNClassifier<T> {
    fn new(k: usize) -> Self {
        Self {
            k,
            ..Default::default()
        }
    }
}

impl<T: Ord + Eq + Clone> KNNClassifier<T> {
    fn test_individual(&self, other: &NewType) -> Option<T> {
        self.data.as_ref().map(|ball| {
            let mut tree = BTreeMap::new();
            for el in ball.query().nn(other).map(|(_, _, n)| n).take(self.k) {
                *tree.entry(el.clone()).or_insert(0) += 1;
            }
            tree.into_iter()
                .max_by_key(|(_, x)| *x)
                .expect("Ball tree is not empty")
                .0
        })
    }
}

#[derive(Debug)]
pub enum ExtendedError {
    InvalidSizes,
    InvalidTrainingSize,
    NumericLabels,
    DataNotPreprocessed,
    ModelNotTrained,
    PolarsError(PolarsError),
}

fn transform_to_vec(data: ndarray::Array2<f64>) -> Vec<NewType> {
    data.axis_iter(Axis(0))
        .map(|el| NewType(el.into_iter().copied().collect_vec()))
        .collect_vec()
}

impl<T: Ord + Eq + Clone> tools::SupervisedLearning for KNNClassifier<T> {
    type Score = MeasurePerformance;
    type Error = ExtendedError;
    type Labels = T;
    fn train(&mut self, x_train: &DataFrame, y: &Vec<T>) -> Result<(), Self::Error> {
        if x_train.shape().0 != y.len() {
            Err(ExtendedError::InvalidSizes)
        } else {
            let data = x_train
                .clone()
                .to_ndarray::<Float64Type>()
                .map_err(ExtendedError::PolarsError)?;

            self.data = Some(BallTree::new(transform_to_vec(data), y.clone()));
            Ok(())
        }
    }

    fn test(&self, x_test: &DataFrame, labels: &Vec<T>) -> Result<Self::Score, Self::Error> {
        let data = x_test
            .to_ndarray::<Float64Type>()
            .map_err(ExtendedError::PolarsError)?;
        let predicted_labels = transform_to_vec(data)
            .into_iter()
            .map(|e| self.test_individual(&e))
            .collect::<Option<Vec<_>>>()
            .ok_or(ExtendedError::ModelNotTrained)?;

        MeasurePerformance::new(&predicted_labels, &labels).ok_or(ExtendedError::InvalidSizes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use itertools::Itertools;

    use ndarray_rand::RandomExt;
    use rand::{distributions::Bernoulli, prelude::Distribution, rngs::StdRng, SeedableRng};
    use std::collections::BTreeSet;
    use tools::SupervisedLearning;

    fn get_distribution(label: &Vec<u8>) -> BTreeSet<(i32, usize)> {
        label
            .iter()
            .copied()
            .map(|x| x as i32)
            .counts()
            .into_iter()
            .collect()
    }

    fn create_data(
        points: usize,
        dims: usize,
        seed: u64,
        noisy: bool,
        limit: f64,
    ) -> (DataFrame, Vec<u8>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let bern = Bernoulli::new(0.05).unwrap();
        let data = Array::random_using(
            (points, dims),
            rand::distributions::Uniform::new(0., 10.),
            &mut rng,
        );
        let example = data
            .sum_axis(Axis(1))
            .map(|x| if *x > limit { 1 } else { 0 })
            .into_iter()
            .map(|x| {
                if noisy && bern.sample(&mut rng) {
                    1 - x
                } else {
                    x
                }
            })
            .collect_vec();

        let mut cols = Vec::with_capacity(dims);
        for (ind, arr) in data.axis_iter(Axis(1)).enumerate() {
            cols.push(Series::new(&ind.to_string(), arr.to_vec()));
        }
        let df = DataFrame::new(cols).expect("Series to dataframe should work");
        (df, example)
    }

    #[test]
    fn create_k_nn() {
        let class = KNNClassifier::<bool>::new(3);
        println!("KNN classifier {:?}", class);
    }

    #[test]
    fn test_items() {
        let (df, labels) = create_data(5, 2, 21, false, 5.);
        let dist_given = get_distribution(&labels);
        println!("{:?}", df);
        let mut kclass = KNNClassifier::default();
        let _ = kclass.train(&df, &labels).expect("Should parse data");

        let labels_dist = kclass
            .data
            .unwrap()
            .query()
            .nn(&NewType(vec![0., 0.]))
            .into_iter()
            .map(|(_, _, a)| *a)
            .collect_vec();
        let label_dist = get_distribution(&labels_dist);

        println!("{:?} {:?}", label_dist, dist_given);
        assert!(
            label_dist.is_subset(&dist_given) && dist_given.is_subset(&label_dist),
            "Labels do not match"
        );
    }

    #[test]
    pub fn test_labels() {
        // Only useful on more complex structures

        let (df, labels) = create_data(5000, 30, 62, false, 20.);
        let mut kclass = KNNClassifier::default();
        kclass.train(&df, &labels).expect("Should parse data");
        let test_accuracy = kclass.test(&df, &labels).unwrap();
        assert_relative_eq!(test_accuracy.accuracy, 1.,);
    }
}
