use std::default;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    SGD(f32),
    Momentum,
    RMSProp,
    Adam,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::SGD(0.001)
    }
}
