#[allow(unused_imports)]
use polars::prelude::*;
// TODO change dataframe type to something more general

pub trait SupervisedLearning {
    type Error;
    type Score;
    type Labels;
    fn train(&mut self, x_train: &DataFrame, y: &Vec<Self::Labels>) -> Result<(), Self::Error>;
    fn test(&self, x_test: &DataFrame, y: &Vec<Self::Labels>) -> Result<Self::Score, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn example() {
        let df = LazyCsvReader::new("../dataset/Cancer_Data.csv")
            .has_header(true)
            .finish()
            .expect("Failed to create dataframe")
            .collect()
            .expect("Faield to convert to dataframe");

        println!("{}", df);
    }
}
