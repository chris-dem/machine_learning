use polars::prelude::*;

mod tree;
/*
Deciison trees are based around using tree like structures in order to iteratively split the data
Data can only be classified using a linear combination of linear classifiers
*/

// Description of things that are needed
// Tree structure in order to store the trees
// Types of data, splitting rules
// Implement the machine learning trait
// Algorithms: Greedy or ID3, will also need to include polars

#[derive(Debug)]
pub struct DecisionTree {}

impl tools::MachineLearning for DecisionTree {
    fn test(&self, _training_data: &DataFrame, _y: Option<&Series>) -> f32 {
        todo!("Training method undefined")
    }
    fn train(&mut self, _x_trait: &DataFrame, _y: Option<&Series>) -> f32 {
        todo!("Testing method undefined")
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
// }
