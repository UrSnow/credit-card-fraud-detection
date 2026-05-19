# Dataset Setup

The raw dataset is not stored in this GitHub repository because it is a large external data file. Download it separately before running the notebooks or `src` scripts.

## Download

Dataset: Credit Card Fraud Detection  
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The downloaded archive should contain a file named `creditcard.csv`.

## Expected Project Location

Create a `data` folder in the project root and place the dataset inside it:

```text
Project/
  data/
    creditcard.csv
```

The default code expects this exact path:

```text
data/creditcard.csv
```

If your downloaded file has a different name, rename it to:

```text
creditcard.csv
```

## Alternative Path

You can also pass a custom path to the preprocessing script:

```bash
python src/preprocessing.py --data-path path/to/your/file.csv
```

The notebooks currently assume the standard location `../data/creditcard.csv`, so using the standard `data/creditcard.csv` location is recommended.

## Git Note

The `data/*.csv` files are intentionally ignored by `.gitignore`. Keep the dataset locally, but do not commit it to GitHub.
