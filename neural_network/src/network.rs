use ndarray::prelude::*;

use crate::{
    activation::Activation, loss_function::TypeFunction, optimizer::Optimizer,
    reguralizer::Reguralizer,
};

pub struct NeuralNetwork {
    pub(crate) weights: Vec<Array2<f64>>,
    pub(crate) activations: Vec<Activation>,
    pub(crate) optimizer: Optimizer,
    pub(crate) reguralization: Option<Reguralizer>,
    pub(crate) loss_func: TypeFunction,
    pub(crate) computed_values: Vec<Vec<f64>>,
}

pub enum ResultFunction {
    Class(usize),
    FLoat(f64),
}

pub enum NNError {
    MismatchedInput,
}

impl NeuralNetwork {
    fn feedforward(&self, input: Array1<f32>) -> Result<ResultFunction, NNError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod init {
        use super::*;
    }
}
