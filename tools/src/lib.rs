#[allow(unused_imports)]
use polars::prelude::*;

pub trait MachineLearning {
    fn train(&mut self, x_train: &DataFrame, y: Option<&Series>) -> f32;
    fn test(&self, training_data: &DataFrame, y: Option<&Series>) -> f32;
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
