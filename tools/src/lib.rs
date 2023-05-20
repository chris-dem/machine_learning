#[allow(unused_imports)]
use polars::prelude::*;

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
