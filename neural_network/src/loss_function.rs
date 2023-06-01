#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClassificationLoss {
    CategoricalCrossEntropy(usize),
    BinaryCrossEntropy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegressionLoss {
    MeanSquaredError,
    AbsoluteError,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TypeFunction {
    Classification(ClassificationLoss),
    Regression(RegressionLoss),
}

impl Default for TypeFunction {
    fn default() -> Self {
        Self::Regression(RegressionLoss::MeanSquaredError)
    }
}
