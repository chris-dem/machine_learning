use itertools::Itertools;
use ndarray::Array2;

use crate::{
    activation::Activation,
    loss_function::{self, ClassificationLoss, TypeFunction},
    network::NeuralNetwork,
    optimizer::Optimizer,
    reguralizer::Reguralizer,
};

use ndarray_rand::{rand_distr::Uniform, RandomExt};

#[derive(Debug, Default)]
pub struct NetworkBuilder {
    pub layers: Option<Vec<usize>>,
    pub activations: Option<Vec<Activation>>,
    pub optimizer: Optimizer,
    pub reguralization: Option<Reguralizer>,
    pub loss_func: TypeFunction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkBuilderError {
    AtLeastOneLayers,
    ActivationsAndLayersDoNotMatch,
    LayerSizeAtLeast1,
    NetworkTopologyUndefined,
}

impl NetworkBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_layers(mut self, layers: &[usize]) -> Self {
        self.layers = Some(layers.iter().copied().collect());
        self
    }

    pub fn set_activations(mut self, acts: &[Activation]) -> Self {
        self.activations = Some(acts.iter().copied().collect());
        self
    }

    pub fn set_optimizer(mut self, opt: Optimizer) -> Self {
        self.optimizer = opt;
        self
    }

    pub fn set_reguralizer(mut self, reg: Reguralizer) -> Self {
        self.reguralization = Some(reg);
        self
    }

    pub fn set_loss_function(mut self, loss: TypeFunction) -> Self {
        self.loss_func = loss;
        self
    }

    pub fn create_network(self) -> Result<NeuralNetwork, NetworkBuilderError> {
        let loss_func = self.loss_func;
        let final_layer_size = match &loss_func {
            TypeFunction::Classification(ClassificationLoss::BinaryCrossEntropy) => 2,
            TypeFunction::Classification(ClassificationLoss::CategoricalCrossEntropy(n)) => *n,
            TypeFunction::Regression(_) => 1,
        };
        let weights_size = self
            .layers
            .ok_or(NetworkBuilderError::NetworkTopologyUndefined)
            .and_then(|x| {
                if x.len() == 0 {
                    Err(NetworkBuilderError::AtLeastOneLayers)
                } else if x.iter().any(|x| *x == 0) {
                    Err(NetworkBuilderError::LayerSizeAtLeast1)
                } else {
                    Ok(x)
                }
            })?
            .into_iter()
            .map(|x| x + 1)
            .chain([final_layer_size])
            .collect_vec();

        let activations = self.activations.unwrap_or_else(|| {
            (0..weights_size.len() - 1)
                .map(|_| Activation::ReLU)
                .collect_vec()
        });
        if activations.len() != weights_size.len() - 1 {
            return Err(NetworkBuilderError::ActivationsAndLayersDoNotMatch);
        }

        let weights = weights_size
            .windows(2)
            .map(|els| Array2::random((els[0], els[1]), Uniform::new(-10., 10.)))
            .collect_vec();

        let computed_values = weights_size
            .iter()
            .map(|layer_len| (0..*layer_len).map(|_| 0.).collect_vec())
            .collect_vec();

        Ok(NeuralNetwork {
            loss_func,
            optimizer: self.optimizer,
            reguralization: self.reguralization,
            activations,
            weights,
            computed_values,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        activation::Activation, loss_function::TypeFunction, optimizer::Optimizer,
        reguralizer::Reguralizer,
    };

    use super::NetworkBuilder;

    #[test]
    pub fn test_builder() {
        let builder = NetworkBuilder::new()
            .set_layers(&[3, 2])
            .set_activations(&[Activation::ReLU, Activation::Sigmoid])
            .set_optimizer(Optimizer::RMSProp)
            .set_reguralizer(Reguralizer::L2)
            .set_loss_function(TypeFunction::Regression(
                crate::loss_function::RegressionLoss::MeanSquaredError,
            ));
        assert!(builder
            .layers
            .as_ref()
            .unwrap()
            .iter()
            .zip([3, 2].iter())
            .all(|(a, b)| a.eq(b)));
        assert!(builder
            .activations
            .as_ref()
            .unwrap()
            .iter()
            .zip([Activation::ReLU, Activation::Sigmoid].iter())
            .all(|(a, b)| a.eq(b)));
        assert!(matches!(builder.optimizer, Optimizer::RMSProp));
        assert!(matches!(
            builder.loss_func,
            TypeFunction::Regression(crate::loss_function::RegressionLoss::MeanSquaredError)
        ));
        assert!(matches!(builder.reguralization, Some(Reguralizer::L2)));
    }

    mod invalid_states {
        use crate::network_builder::NetworkBuilderError;

        use super::*;

        #[test]
        fn test_invalid_layers() {
            let builder = NetworkBuilder::default()
                .set_layers(&[])
                .set_activations(&[]);

            assert_eq!(
                builder.create_network().err().unwrap(),
                NetworkBuilderError::AtLeastOneLayers
            );
        }

        #[test]
        fn test_mismatched_activation() {
            let builder = NetworkBuilder::new()
                .set_layers(&[3, 2])
                .set_activations(&[Activation::ReLU]);
            assert_eq!(
                builder.create_network().err().unwrap(),
                NetworkBuilderError::ActivationsAndLayersDoNotMatch
            );
        }

        #[test]
        fn test_layers_at_least_one() {
            let builder = NetworkBuilder::new().set_layers(&[0, 2]);

            assert_eq!(
                builder.create_network().err().unwrap(),
                NetworkBuilderError::LayerSizeAtLeast1
            );
        }
    }

    mod test_neural_network_creation {
        use super::*;
        
    }
}
