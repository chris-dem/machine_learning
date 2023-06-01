#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLU,           // Leaky
    SELU(f32, f32), // Scaled exponential
    PRELU(f32),     // Scaled leaky
}
