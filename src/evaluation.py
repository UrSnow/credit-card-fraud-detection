
"""Evaluation helpers for supervised and anomaly-detection models."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _confusion_counts(y_true, y_pred):
    """Return confusion-matrix counts as TN, FP, FN, TP."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def evaluate_binary_scores(model_name, y_true, y_score, threshold):
    """Evaluate probability-like scores for a binary classifier."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)

    return {
        "Model": model_name,
        "Threshold": threshold,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "PR_AUC": average_precision_score(y_true, y_score),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def evaluate_anomaly_scores(model_name, y_true, anomaly_score, threshold, percentile=None):
    """Evaluate anomaly scores where larger values indicate higher anomaly risk."""
    y_pred = (np.asarray(anomaly_score) > threshold).astype(int)
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)

    return {
        "Model": model_name,
        "Percentile": percentile,
        "Threshold": threshold,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, anomaly_score),
        "PR_AUC": average_precision_score(y_true, anomaly_score),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def tune_binary_thresholds(
    model_name,
    y_true,
    y_score,
    thresholds=DEFAULT_THRESHOLD_GRID,
):
    """Evaluate a binary classifier over a threshold grid."""
    rows = [
        evaluate_binary_scores(model_name, y_true, y_score, threshold)
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def tune_anomaly_thresholds(
    model_name,
    y_true,
    anomaly_score,
    threshold_values,
    percentiles=None,
):
    """Evaluate anomaly scores over explicit thresholds."""
    if percentiles is None:
        percentiles = [None] * len(threshold_values)

    rows = [
        evaluate_anomaly_scores(
            model_name,
            y_true,
            anomaly_score,
            threshold,
            percentile=percentile,
        )
        for threshold, percentile in zip(threshold_values, percentiles)
    ]
    return pd.DataFrame(rows)


def select_best_result(results_df, sort_by=("F1", "PR_AUC", "Recall")):
    """Select the best row from a results dataframe using descending metrics."""
    if results_df.empty:
        raise ValueError("results_df is empty.")
    return results_df.sort_values(list(sort_by), ascending=False).iloc[0]


def classification_report_at_threshold(y_true, y_score, threshold, anomaly=False):
    """Return a text classification report at the selected threshold."""
    if anomaly:
        y_pred = (np.asarray(y_score) > threshold).astype(int)
    else:
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return classification_report(y_true, y_pred, digits=4, zero_division=0)


def confusion_matrix_at_threshold(y_true, y_score, threshold, anomaly=False):
    """Return a confusion matrix at the selected threshold."""
    if anomaly:
        y_pred = (np.asarray(y_score) > threshold).astype(int)
    else:
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def load_final_results(results_dir):
    """Load saved final test results if they exist."""
    results_dir = Path(results_dir)
    frames = []

    baseline_path = results_dir / "baseline_test_results.csv"
    mlp_path = results_dir / "mlp_test_results.csv"
    autoencoder_path = results_dir / "autoencoder_test_results.csv"

    if baseline_path.exists():
        frames.append(pd.read_csv(baseline_path))
    if mlp_path.exists():
        frames.append(pd.read_csv(mlp_path))
    if autoencoder_path.exists():
        frames.append(pd.read_csv(autoencoder_path))

    if not frames:
        raise FileNotFoundError(
            f"No final result CSV files found in {results_dir}. "
            "Run src/models.py first."
        )

    return pd.concat(frames, ignore_index=True, sort=False)


def _safe_filename(name):
    """Create a simple filesystem-safe name for generated figures."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def _final_model_labels(results_df):
    """Return compact labels for final-model comparison figures."""
    labels = results_df["Model"].replace(
        {
            "Random Forest - balanced": "Random Forest",
            "AE_bottleneck_6_dropout_0.2": "Autoencoder",
        }
    )
    return labels.map(lambda label: "MLP" if str(label).startswith("MLP_") else label)


def save_roc_pr_curves(y_true, y_score, model_name, figure_dir, filename_prefix):
    """Save ROC and precision-recall curves."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(fpr, tpr)
    axes[0].set_title(f"ROC Curve - {model_name}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[1].plot(recall, precision)
    axes[1].set_title(f"Precision-Recall Curve - {model_name}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    fig.tight_layout()
    output_path = figure_dir / f"{filename_prefix}_roc_pr_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_confusion_matrix_plot(
    y_true,
    y_score,
    threshold,
    model_name,
    figure_dir,
    filename_prefix,
    anomaly=False,
):
    """Save a labelled confusion matrix for a selected threshold."""
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if anomaly:
        y_pred = (np.asarray(y_score) > threshold).astype(int)
    else:
        y_pred = (np.asarray(y_score) >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_xticks([0, 1], ["Normal", "Fraud"])
    ax.set_yticks([0, 1], ["Normal", "Fraud"])

    threshold_for_text = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            text_color = "white" if value > threshold_for_text else "black"
            ax.text(
                column,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = figure_dir / f"{filename_prefix}_confusion_matrix.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metric_comparison_plot(results_df, figure_dir):
    """Save a grouped bar chart comparing final model metrics."""
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    available_metrics = [metric for metric in metrics if metric in results_df.columns]
    model_labels = _final_model_labels(results_df)

    x = np.arange(len(available_metrics))
    width = 0.35 if len(results_df) <= 2 else 0.8 / len(results_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    for index, (_, row) in enumerate(results_df.iterrows()):
        offset = (index - (len(results_df) - 1) / 2) * width
        values = [row[metric] for metric in available_metrics]
        bars = ax.bar(x + offset, values, width, label=model_labels.iloc[index])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Final Test Metric Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x, available_metrics)
    ax.set_ylim(0, 1.08)
    ax.legend()
    fig.tight_layout()
    output_path = figure_dir / "model_final_metric_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_error_count_comparison_plot(results_df, figure_dir):
    """Save FP/FN/TP count comparison for the final selected models."""
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    counts = ["FP", "FN", "TP"]
    available_counts = [count for count in counts if count in results_df.columns]
    model_labels = _final_model_labels(results_df)

    x = np.arange(len(available_counts))
    width = 0.35 if len(results_df) <= 2 else 0.8 / len(results_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    for index, (_, row) in enumerate(results_df.iterrows()):
        offset = (index - (len(results_df) - 1) / 2) * width
        values = [int(row[count]) for count in available_counts]
        bars = ax.bar(x + offset, values, width, label=model_labels.iloc[index])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(value, 1) * 1.08,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Final Test Detection Count Comparison")
    ax.set_xlabel("Count Type")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_yscale("log")
    ax.set_xticks(x, available_counts)
    ax.legend()
    fig.tight_layout()
    output_path = figure_dir / "model_final_detection_count_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_reconstruction_error_histogram(
    y_true,
    reconstruction_errors,
    threshold,
    model_name,
    figure_dir,
    x_limit_percentile=99.5,
):
    """Save autoencoder reconstruction-error distribution.

    The x-axis is zoomed by default because a few extreme reconstruction-error
    outliers can make the main normal/fraud overlap unreadable.
    """
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    reconstruction_errors = np.asarray(reconstruction_errors)

    visible_limit = np.percentile(reconstruction_errors, x_limit_percentile)
    bins = np.linspace(0, visible_limit, 80)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        reconstruction_errors[y_true == 0],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Normal",
    )
    ax.hist(
        reconstruction_errors[y_true == 1],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label="Fraud",
    )
    ax.axvline(threshold, linestyle="--", color="black", label="Selected Threshold")
    ax.set_title(f"Reconstruction Error Distribution - {model_name}")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(0, visible_limit)
    ax.text(
        0.02,
        0.88,
        f"Density plot; x-axis clipped at {x_limit_percentile}th percentile",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    fig.tight_layout()
    output_path = figure_dir / f"{_safe_filename(model_name)}_reconstruction_errors.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def reconstruction_error(model, X):
    """Compute row-wise mean squared reconstruction error."""
    reconstructions = model.predict(X, verbose=0)
    X_values = X.values if hasattr(X, "values") else np.asarray(X)
    return np.mean(np.square(X_values - reconstructions), axis=1)


def save_final_result_plots(
    data_path="outputs/results_corrected/creditcard_cleaned.pkl",
    results_dir="outputs/results_corrected",
    figures_dir="outputs/figures",
):
    """Regenerate final plots from saved models and processed data."""
    data_path = Path(data_path)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    data = joblib.load(data_path)
    saved_paths = []

    baseline_model_path = results_dir / "baseline_models.pkl"
    if baseline_model_path.exists():
        baseline_bundle = joblib.load(baseline_model_path)
        best_name = baseline_bundle["best_model_name"]
        best_threshold = float(baseline_bundle["best_threshold"])
        best_model = baseline_bundle["trained_models"][best_name]
        test_scores = best_model.predict_proba(data["X_test"])[:, 1]
        saved_paths.append(
            save_roc_pr_curves(
                data["y_test"],
                test_scores,
                best_name,
                figures_dir,
                "supervised_selected_model",
            )
        )
        saved_paths.append(
            save_confusion_matrix_plot(
                data["y_test"],
                test_scores,
                best_threshold,
                best_name,
                figures_dir,
                "supervised_selected_model",
            )
        )

    mlp_model_path = results_dir / "mlp_best_model.keras"
    mlp_selection_path = results_dir / "mlp_selection.pkl"
    if mlp_model_path.exists() and mlp_selection_path.exists():
        from tensorflow.keras.models import load_model

        selection = joblib.load(mlp_selection_path)
        best_name = selection["best_mlp_name"]
        best_threshold = float(selection["best_mlp_threshold"])
        mlp_model = load_model(mlp_model_path)
        test_scores = mlp_model.predict(data["X_test"], verbose=0).reshape(-1)

        saved_paths.append(
            save_roc_pr_curves(
                data["y_test"],
                test_scores,
                best_name,
                figures_dir,
                "mlp_selected_model",
            )
        )
        saved_paths.append(
            save_confusion_matrix_plot(
                data["y_test"],
                test_scores,
                best_threshold,
                best_name,
                figures_dir,
                "mlp_selected_model",
            )
        )

    autoencoder_model_path = results_dir / "autoencoder_best_model.keras"
    autoencoder_selection_path = results_dir / "autoencoder_selection.pkl"
    if autoencoder_model_path.exists() and autoencoder_selection_path.exists():
        from tensorflow.keras.models import load_model

        selection = joblib.load(autoencoder_selection_path)
        best_name = selection["best_autoencoder_name"]
        best_threshold = float(selection["best_autoencoder_threshold"])
        autoencoder = load_model(autoencoder_model_path)
        test_loss = reconstruction_error(autoencoder, data["X_test"])

        saved_paths.append(
            save_reconstruction_error_histogram(
                data["y_test"],
                test_loss,
                best_threshold,
                best_name,
                figures_dir,
            )
        )
        saved_paths.append(
            save_roc_pr_curves(
                data["y_test"],
                test_loss,
                best_name,
                figures_dir,
                f"{_safe_filename(best_name)}_test",
            )
        )
        saved_paths.append(
            save_confusion_matrix_plot(
                data["y_test"],
                test_loss,
                best_threshold,
                best_name,
                figures_dir,
                f"{_safe_filename(best_name)}_test",
                anomaly=True,
            )
        )

    final_results = load_final_results(results_dir)
    saved_paths.append(save_metric_comparison_plot(final_results, figures_dir))
    saved_paths.append(save_error_count_comparison_plot(final_results, figures_dir))

    return saved_paths


def main():
    """Print final saved model results from the command line."""
    parser = argparse.ArgumentParser(description="Show final saved model results.")
    parser.add_argument("--results-dir", default="outputs/results_corrected")
    parser.add_argument(
        "--data-path",
        default="outputs/results_corrected/creditcard_cleaned.pkl",
    )
    parser.add_argument("--figures-dir", default="outputs/figures")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only print results; do not regenerate final plots.",
    )
    args = parser.parse_args()

    results = load_final_results(args.results_dir)
    columns = [
        "Model",
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    columns = [column for column in columns if column in results.columns]

    print(results[columns].to_string(index=False))

    if not args.no_plots:
        saved_paths = save_final_result_plots(
            data_path=args.data_path,
            results_dir=args.results_dir,
            figures_dir=args.figures_dir,
        )
        if saved_paths:
            print("\nSaved figures:")
            for path in saved_paths:
                print(path)
        else:
            print("\nNo saved models found for plot generation.")


if __name__ == "__main__":
    main()
