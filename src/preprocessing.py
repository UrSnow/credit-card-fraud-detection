"""Preprocessing helpers for the ACML credit card fraud project."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 42
DEFAULT_TARGET_COLUMN = "Class"
DEFAULT_COLUMNS_TO_SCALE = ("Time", "Amount")


def load_dataset(data_path):
    """Load the raw credit card fraud dataset."""
    return pd.read_csv(Path(data_path))


def remove_duplicates(df, keep="first"):
    """Return a copy of the dataset with duplicate rows removed."""
    return df.drop_duplicates(keep=keep).copy()


def split_features_target(df, target_column=DEFAULT_TARGET_COLUMN):
    """Split a dataframe into features and target."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def stratified_train_val_test_split(
    X,
    y,
    test_size=0.15,
    val_size=0.15,
    random_state=DEFAULT_RANDOM_STATE,
):
    """Create a stratified train/validation/test split.

    `test_size` and `val_size` are fractions of the full dataset.  70/15/15 split.
    """
    if test_size <= 0 or val_size <= 0:
        raise ValueError("test_size and val_size must be positive.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    validation_fraction = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y_train_full,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_selected_columns(
    X_train,
    X_val,
    X_test,
    columns_to_scale=DEFAULT_COLUMNS_TO_SCALE,
):
    """Fit a StandardScaler on training columns and transform all splits."""
    scaler = StandardScaler()
    columns_to_scale = list(columns_to_scale)

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[columns_to_scale] = scaler.fit_transform(
        X_train_scaled[columns_to_scale]
    )
    X_val_scaled[columns_to_scale] = scaler.transform(X_val_scaled[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test_scaled[columns_to_scale])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def class_distribution(*named_targets):
    """Return class percentages for named target splits.

    Pass tuples such as ("train", y_train).
    """
    distributions = {}
    for name, y in named_targets:
        distributions[name] = y.value_counts(normalize=True).sort_index() * 100
    return pd.DataFrame(distributions)


def _as_dataframe(data):
    """Return a dataframe from either an existing dataframe or a CSV path."""
    if isinstance(data, pd.DataFrame):
        return data
    return load_dataset(data)


def _prepare_figure_dir(figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _save_figure(fig, figures_dir, filename):
    """Save and close a matplotlib figure."""
    import matplotlib.pyplot as plt

    figures_dir = _prepare_figure_dir(figures_dir)
    output_path = figures_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _add_bar_labels(ax, bars, labels):
    """Add compact labels above bars, including on log-scaled axes."""
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 1.04,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_class_distribution_plot(
    data,
    figures_dir="outputs/figures",
    target_column=DEFAULT_TARGET_COLUMN,
):
    """Save the raw class distribution plot before cleaning."""
    import matplotlib.pyplot as plt

    df = _as_dataframe(data)
    counts = df[target_column].value_counts().sort_index()
    labels = ["Normal" if int(label) == 0 else "Fraud" for label in counts.index]
    percentages = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["#2563eb", "#f97316"])
    ax.set_yscale("log")
    ax.set_ylim(max(1, counts.min() * 0.6), counts.max() * 2.2)
    ax.set_title("Class Distribution Before Cleaning")
    ax.set_xlabel("Class")
    ax.set_ylabel("Transaction Count (log scale)")
    _add_bar_labels(
        ax,
        bars,
        [
            f"{count:,.0f}\n({percentage:.3f}%)"
            for count, percentage in zip(counts.values, percentages.values)
        ],
    )
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "data_class_distribution_before_cleaning.png")


def save_duplicate_summary_plot(
    raw_data,
    cleaned_data=None,
    figures_dir="outputs/figures",
):
    """Save a summary plot showing rows removed by duplicate cleaning."""
    import matplotlib.pyplot as plt

    raw_df = _as_dataframe(raw_data)
    duplicate_count = int(raw_df.duplicated().sum())
    if cleaned_data is None:
        cleaned_rows = len(raw_df) - duplicate_count
    else:
        cleaned_rows = len(_as_dataframe(cleaned_data))

    categories = ["Rows Before", "Duplicates Removed", "Rows After"]
    values = [len(raw_df), duplicate_count, cleaned_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(categories, values, color=["#2563eb", "#dc2626", "#16a34a"])
    ax.set_yscale("log")
    positive_values = [value for value in values if value > 0]
    ax.set_ylim(max(1, min(positive_values) * 0.6), max(values) * 2.2)
    ax.set_title("Duplicate Removal Summary")
    ax.set_ylabel("Rows (log scale)")
    _add_bar_labels(ax, bars, [f"{value:,.0f}" for value in values])
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "preprocessing_duplicate_rows_summary.png")


def save_amount_distribution_plot(
    data,
    figures_dir="outputs/figures",
    amount_column="Amount",
):
    """Save the transaction amount histogram before scaling."""
    import matplotlib.pyplot as plt

    df = _as_dataframe(data)
    amount = pd.to_numeric(df[amount_column], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(amount, bins=80, color="#2563eb", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Transaction Amount Distribution Before Scaling")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency (log scale)")
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "data_amount_distribution_before_scaling.png")


def save_amount_by_class_boxplot(
    data,
    figures_dir="outputs/figures",
    target_column=DEFAULT_TARGET_COLUMN,
    amount_column="Amount",
):
    """Save a readable amount-by-class boxplot using log1p amount values."""
    import matplotlib.pyplot as plt

    df = _as_dataframe(data)
    normal_amount = np.log1p(df.loc[df[target_column] == 0, amount_column])
    fraud_amount = np.log1p(df.loc[df[target_column] == 1, amount_column])

    fig, ax = plt.subplots(figsize=(6, 4))
    box = ax.boxplot(
        [normal_amount, fraud_amount],
        tick_labels=["Normal", "Fraud"],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], ["#93c5fd", "#fdba74"]):
        patch.set_facecolor(color)

    ax.set_title("Transaction Amount by Class Before Scaling")
    ax.set_xlabel("Class")
    ax.set_ylabel("log1p(Amount)")
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "data_amount_by_class_boxplot.png")


def save_time_distribution_by_class_plot(
    data,
    figures_dir="outputs/figures",
    target_column=DEFAULT_TARGET_COLUMN,
    time_column="Time",
):
    """Save the transaction-time distribution by class."""
    import matplotlib.pyplot as plt

    df = _as_dataframe(data)
    fig, ax = plt.subplots(figsize=(8, 4))

    class_styles = [(0, "Normal", "#2563eb"), (1, "Fraud", "#f97316")]
    for class_value, label, color in class_styles:
        hours = df.loc[df[target_column] == class_value, time_column] / 3600
        ax.hist(
            hours,
            bins=48,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
            color=color,
        )

    ax.set_title("Transaction Time Distribution by Class")
    ax.set_xlabel("Hours Since First Transaction")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, figures_dir, "data_time_distribution_by_class.png")


def save_feature_correlation_heatmap(
    data,
    figures_dir="outputs/figures",
    target_column=DEFAULT_TARGET_COLUMN,
):
    """Save a feature correlation heatmap after duplicate removal.

    This is a normalized covariance view. It is more readable than raw covariance
    because the dataset mixes PCA features with the larger-scale Time and Amount
    columns.
    """
    import matplotlib.pyplot as plt

    df = remove_duplicates(_as_dataframe(data))
    feature_df = df.drop(columns=[target_column])
    correlation = feature_df.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix After Duplicate Removal")
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_yticks(np.arange(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(correlation.columns, fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")

    fig.tight_layout()
    return _save_figure(fig, figures_dir, "data_feature_correlation_matrix_after_cleaning.png")


def save_split_class_distribution_plot(
    processed_data,
    figures_dir="outputs/figures",
):
    """Save class-count and fraud-rate plots for the train/validation/test split."""
    import matplotlib.pyplot as plt

    split_targets = {
        "Train": processed_data["y_train"],
        "Validation": processed_data["y_val"],
        "Test": processed_data["y_test"],
    }

    rows = []
    for split_name, y in split_targets.items():
        counts = y.value_counts().sort_index()
        normal_count = int(counts.get(0, 0))
        fraud_count = int(counts.get(1, 0))
        total = normal_count + fraud_count
        rows.append(
            {
                "Split": split_name,
                "Normal": normal_count,
                "Fraud": fraud_count,
                "Fraud Rate (%)": fraud_count / total * 100,
            }
        )

    split_df = pd.DataFrame(rows)
    x = np.arange(len(split_df))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    normal_bars = axes[0].bar(
        x - width / 2,
        split_df["Normal"],
        width,
        label="Normal",
        color="#2563eb",
    )
    fraud_bars = axes[0].bar(
        x + width / 2,
        split_df["Fraud"],
        width,
        label="Fraud",
        color="#f97316",
    )
    axes[0].set_yscale("log")
    count_values = split_df[["Normal", "Fraud"]].to_numpy().ravel()
    axes[0].set_ylim(max(1, count_values.min() * 0.6), count_values.max() * 2.2)
    axes[0].set_title("Class Counts by Data Split")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Transactions (log scale)")
    axes[0].set_xticks(x, split_df["Split"])
    axes[0].legend()
    _add_bar_labels(
        axes[0],
        normal_bars,
        [f"{value:,.0f}" for value in split_df["Normal"]],
    )
    _add_bar_labels(
        axes[0],
        fraud_bars,
        [f"{value:,.0f}" for value in split_df["Fraud"]],
    )

    rate_bars = axes[1].bar(
        split_df["Split"],
        split_df["Fraud Rate (%)"],
        color="#16a34a",
    )
    axes[1].set_title("Fraud Rate Preserved by Stratified Split")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].set_ylim(0, max(split_df["Fraud Rate (%)"]) * 1.35)
    _add_bar_labels(
        axes[1],
        rate_bars,
        [f"{value:.3f}%" for value in split_df["Fraud Rate (%)"]],
    )

    fig.tight_layout()
    return _save_figure(fig, figures_dir, "preprocessing_split_class_distribution.png")


def save_preprocessing_figures(
    data_path="data/creditcard.csv",
    processed_data=None,
    processed_data_path=None,
    figures_dir="outputs/figures",
):
    """Save focused data-understanding and preprocessing figures for the report."""
    raw_df = load_dataset(data_path)
    cleaned_df = remove_duplicates(raw_df)

    saved_paths = [
        save_class_distribution_plot(raw_df, figures_dir),
        save_duplicate_summary_plot(raw_df, cleaned_df, figures_dir),
        save_amount_distribution_plot(raw_df, figures_dir),
        save_amount_by_class_boxplot(raw_df, figures_dir),
        save_time_distribution_by_class_plot(raw_df, figures_dir),
        save_feature_correlation_heatmap(cleaned_df, figures_dir),
    ]

    if processed_data is None and processed_data_path is not None:
        processed_data = joblib.load(processed_data_path)

    if processed_data is not None:
        saved_paths.append(save_split_class_distribution_plot(processed_data, figures_dir))

    return saved_paths


def preprocess_creditcard_dataset(
    data_path,
    output_path=None,
    target_column=DEFAULT_TARGET_COLUMN,
    columns_to_scale=DEFAULT_COLUMNS_TO_SCALE,
    test_size=0.15,
    val_size=0.15,
    random_state=DEFAULT_RANDOM_STATE,
):
    """Run the full corrected preprocessing workflow.

    Returns a dictionary containing scaled train/validation/test splits and the
    fitted scaler. The dictionary is saved with joblib in `output_path`.
    """
    df = load_dataset(data_path)
    df = remove_duplicates(df)
    X, y = split_features_target(df, target_column=target_column)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    X_train, X_val, X_test, scaler = scale_selected_columns(
        X_train,
        X_val,
        X_test,
        columns_to_scale=columns_to_scale,
    )

    processed = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "columns_to_scale": list(columns_to_scale),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(processed, output_path)

    return processed


def main():
    """Run the corrected preprocessing pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Preprocess the credit card dataset.")
    parser.add_argument("--data-path", default="data/creditcard.csv")
    parser.add_argument(
        "--output-path",
        default="outputs/results_corrected/creditcard_cleaned.pkl",
    )
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Only preprocess the data; do not save preprocessing figures.",
    )
    args = parser.parse_args()

    processed = preprocess_creditcard_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
    )

    print("Preprocessing complete.")
    print("Saved:", args.output_path)
    print("X_train:", processed["X_train"].shape)
    print("X_val:", processed["X_val"].shape)
    print("X_test:", processed["X_test"].shape)

    if not args.no_figures:
        saved_paths = save_preprocessing_figures(
            data_path=args.data_path,
            processed_data=processed,
            figures_dir=args.figures_dir,
        )
        print("\nSaved preprocessing figures:")
        for path in saved_paths:
            print(path)


if __name__ == "__main__":
    main()
