# ACML Credit Card Fraud Detection Project

This repository contains the notebook experiments, corrected workflow, saved results, reusable source code, and figures for the ACML machine learning project.

## Project Structure

- `notebooks/`: first draft notebooks used for early data understanding, preprocessing, and modelling.
- `notebooks_corrections/`: corrected notebooks with the final train/validation/test methodology.
- `src/`: reusable Python code that reproduces the corrected workflow.
- `outputs/results/`: draft outputs from the original notebooks.
- `outputs/results_corrected/`: final corrected model and metric artifacts.
- `outputs/figures/`: final plots for the report.

Use `notebooks_corrections`, `outputs/results_corrected`, `outputs/figures`, and `src` as the final evidence for the report. The original notebooks and `outputs/results` are draft work.

## Dataset Source

This project uses the public Credit Card Fraud Detection dataset, commonly distributed on Kaggle and originally associated with the Machine Learning Group at Universite Libre de Bruxelles.

Dataset URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset is anonymised. Features `V1` to `V28` are PCA-transformed for confidentiality, while `Time`, `Amount`, and `Class` remain available. No ethics clearance is required for this project because the dataset is public and anonymised.

## What We Created vs External Sources

The dataset comes from the public Kaggle/ULB credit card fraud dataset. The modelling workflow, preprocessing decisions, validation/test split, threshold tuning, model comparison, autoencoder experiments, evaluation analysis, and project code were created for this ACML project.

External Python libraries such as pandas, scikit-learn, matplotlib, joblib, and TensorFlow/Keras are used for data handling, modelling, plotting, saving artifacts, and neural-network implementation.

## Dependencies

Use the same Python environment that was used to run the corrected notebooks. If imports fail, install the pinned dependencies:

```bash
source .venv/Scripts/activate
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Run the Source Code

Run commands from the project root.

### 1. Preprocess the Dataset

This creates the corrected train/validation/test split and saves it to `outputs/results_corrected/creditcard_cleaned.pkl`. It also saves focused data-understanding and preprocessing figures to `outputs/figures`.

```powershell
python src/preprocessing.py
```

Git Bash alternative:

```bash
./.venv/Scripts/python.exe src/preprocessing.py
```

### 2. Train and Evaluate Supervised Models

This trains the corrected supervised candidates, selects the best model and threshold on validation data, then evaluates once on the test set. It can take several minutes because it trains Random Forest and Gradient Boosting candidates.

```powershell
python src/models.py supervised
```

Git Bash alternative:

```bash
./.venv/Scripts/python.exe src/models.py supervised
```

### 3. Train and Evaluate Autoencoders

This trains the autoencoder experiments, selects the best architecture and threshold on validation data, then evaluates once on the test set. It also saves the selected autoencoder training curve to `outputs/figures`.

```powershell
python src/models.py autoencoder
```

Git Bash alternative:

```bash
./.venv/Scripts/python.exe src/models.py autoencoder
```

To run both model pipelines in sequence:

```powershell
python src/models.py all
```

### 4. Show Final Saved Results and Regenerate Figures

This prints the saved final test results and regenerates the final plots in `outputs/figures`.

```powershell
python src/evaluation.py
```

To print results without regenerating plots:

```powershell
python src/evaluation.py --no-plots
```

## Report Figures

The source workflow saves both preprocessing and model-evaluation figures in `outputs/figures`.

### Preprocessing and Data-Understanding Figures

- `data_class_distribution_before_cleaning.png`  
  Shows the normal/fraud class counts before cleaning. The dataset is extremely imbalanced, so accuracy is not a suitable main metric.

- `preprocessing_duplicate_rows_summary.png`  
  Shows rows before cleaning, duplicate rows removed, and rows after cleaning. This documents the main cleaning step before splitting the data.

- `data_amount_distribution_before_scaling.png`  
  Shows the raw transaction amount distribution before scaling. Amount is highly skewed, which supports scaling `Amount` before modelling.

- `data_amount_by_class_boxplot.png`  
  Compares transaction amount by class using `log1p(Amount)`. It shows that amount distributions differ between normal and fraud transactions, but not cleanly enough to solve the problem alone.

- `data_time_distribution_by_class.png`  
  Shows transaction timing patterns for normal and fraud classes. Time may contain useful structure, but it is not by itself a reliable fraud separator.

- `data_feature_correlation_matrix_after_cleaning.png`  
  Shows correlations between numerical features after duplicate removal. Most PCA features have low pairwise correlation, which is expected because they are transformed components.

- `preprocessing_split_class_distribution.png`  
  Shows class counts and fraud rate across train, validation, and test sets. The fraud ratio is preserved, confirming that the split was stratified.

### Model and Result Figures

- `supervised_selected_model_roc_pr_curves.png`  
  Shows ROC and precision-recall curves for the selected Random Forest. The PR curve is especially important because fraud is rare.

- `supervised_selected_model_confusion_matrix.png`  
  Shows final test predictions for the selected Random Forest. It has very few false positives while still detecting most fraud cases.

- `ae_bottleneck_6_dropout_0_2_training_curve.png`  
  Shows autoencoder training and validation loss over epochs. This documents neural-network training behaviour and checks that training converged.

- `ae_bottleneck_6_dropout_0_2_reconstruction_errors.png`  
  Shows reconstruction-error distributions for normal and fraud transactions. The threshold separates some fraud cases, but there is still overlap between classes.

- `ae_bottleneck_6_dropout_0_2_test_roc_pr_curves.png`  
  Shows ROC and precision-recall curves for the autoencoder anomaly scores. ROC-AUC is strong, but PR-AUC is much weaker than Random Forest.

- `ae_bottleneck_6_dropout_0_2_test_confusion_matrix.png`  
  Shows final test predictions for the selected autoencoder. It catches slightly more fraud cases than Random Forest, but creates many more false positives.

- `model_final_metric_comparison.png`  
  Compares Precision, Recall, F1, ROC-AUC, and PR-AUC for the final models. Random Forest is the best practical model because it has much better precision, F1, and PR-AUC.

- `model_final_detection_count_comparison.png`  
  Compares false positives, false negatives, and true positives. This makes the trade-off clear: the autoencoder gains a few true positives but at a large false-positive cost.

## Recommended Run Order

1. `python src/preprocessing.py`
2. `python src/models.py supervised`
3. `python src/models.py autoencoder`
4. `python src/evaluation.py`
