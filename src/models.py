"""Model helpers for the ACML credit card fraud project."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

try:
    from .evaluation import (
        DEFAULT_THRESHOLD_GRID,
        evaluate_anomaly_scores,
        evaluate_binary_scores,
        select_best_result,
        tune_binary_thresholds,
    )
except ImportError:
    from evaluation import (
        DEFAULT_THRESHOLD_GRID,
        evaluate_anomaly_scores,
        evaluate_binary_scores,
        select_best_result,
        tune_binary_thresholds,
    )


DEFAULT_RANDOM_STATE = 42
DEFAULT_AUTOENCODER_CONFIGS = (
    {"name": "AE_bottleneck_4_dropout_0.1", "bottleneck_size": 4, "dropout_rate": 0.1},
    {"name": "AE_bottleneck_6_dropout_0.2", "bottleneck_size": 6, "dropout_rate": 0.2},
    {"name": "AE_bottleneck_8_dropout_0.2", "bottleneck_size": 8, "dropout_rate": 0.2},
    {
        "name": "AE_bottleneck_12_dropout_0.3",
        "bottleneck_size": 12,
        "dropout_rate": 0.3,
    },
)
DEFAULT_AUTOENCODER_LR_SWEEP_CONFIGS = (
    {
        "name": "AE_bottleneck_6_dropout_0.2_lr_1e-4",
        "bottleneck_size": 6,
        "dropout_rate": 0.2,
        "learning_rate": 1e-4,
    },
    {
        "name": "AE_bottleneck_6_dropout_0.2_lr_5e-4",
        "bottleneck_size": 6,
        "dropout_rate": 0.2,
        "learning_rate": 5e-4,
    },
    {
        "name": "AE_bottleneck_6_dropout_0.2_lr_1e-3",
        "bottleneck_size": 6,
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
    },
)
DEFAULT_MLP_CONFIGS = (
    {
        "name": "MLP_dense_64_32_dropout_0.2_lr_1e-3",
        "hidden_units": (64, 32),
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
    },
    {
        "name": "MLP_dense_128_64_32_dropout_0.3_lr_1e-3",
        "hidden_units": (128, 64, 32),
        "dropout_rate": 0.3,
        "learning_rate": 1e-3,
    },
    {
        "name": "MLP_dense_128_64_32_dropout_0.3_lr_5e-4",
        "hidden_units": (128, 64, 32),
        "dropout_rate": 0.3,
        "learning_rate": 5e-4,
    },
)


def _safe_filename(name):
    """Create a simple filesystem-safe name for generated figures."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
    )


def get_supervised_candidate_models(random_state=DEFAULT_RANDOM_STATE):
    """Return the corrected supervised model candidates from notebook 03."""
    return {
        "Logistic Regression - balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest - balanced": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Random Forest - shallow": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting - default": GradientBoostingClassifier(
            random_state=random_state,
        ),
        "Gradient Boosting - tuned": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }


def train_supervised_models(candidate_models, X_train, y_train):
    """Fit all candidate supervised models."""
    trained_models = {}
    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
    return trained_models


def predict_positive_class_scores(trained_models, X):
    """Return positive-class probabilities for fitted sklearn classifiers."""
    return {
        model_name: model.predict_proba(X)[:, 1]
        for model_name, model in trained_models.items()
    }


def evaluate_supervised_default_thresholds(
    trained_models,
    X_val,
    y_val,
    threshold=0.5,
):
    """Evaluate all supervised models at one threshold."""
    scores = predict_positive_class_scores(trained_models, X_val)
    rows = [
        evaluate_binary_scores(model_name, y_val, y_score, threshold)
        for model_name, y_score in scores.items()
    ]
    return pd.DataFrame(rows), scores


def tune_supervised_models(
    validation_scores,
    y_val,
    thresholds=DEFAULT_THRESHOLD_GRID,
):
    """Tune thresholds for all supervised validation scores."""
    threshold_frames = [
        tune_binary_thresholds(model_name, y_val, y_score, thresholds)
        for model_name, y_score in validation_scores.items()
    ]
    return pd.concat(threshold_frames, ignore_index=True)


def select_supervised_model(
    trained_models,
    validation_threshold_df,
    sort_by=("F1", "PR_AUC", "Recall"),
):
    """Select the best supervised model and threshold from validation results."""
    best_row = select_best_result(validation_threshold_df, sort_by=sort_by)
    model_name = best_row["Model"]
    threshold = float(best_row["Threshold"])
    return trained_models[model_name], model_name, threshold, best_row


def build_autoencoder(input_dim, bottleneck_size=6, dropout_rate=0.2, learning_rate=1e-3):
    """Build the dense autoencoder architecture from notebook 04."""
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Model

    input_layer = Input(shape=(input_dim,))

    encoder = Dense(24, activation="relu")(input_layer)
    encoder = Dropout(dropout_rate)(encoder)
    encoder = Dense(12, activation="relu")(encoder)
    bottleneck = Dense(bottleneck_size, activation="relu")(encoder)

    decoder = Dense(12, activation="relu")(bottleneck)
    decoder = Dropout(dropout_rate)(decoder)
    decoder = Dense(24, activation="relu")(decoder)
    output_layer = Dense(input_dim, activation="linear")(decoder)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def build_mlp_classifier(input_dim, hidden_units, dropout_rate, learning_rate):
    """Build the dense supervised MLP classifier from notebook 05."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    model = Sequential(name="fraud_mlp")
    model.add(Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def compute_balanced_class_weights(y):
    """Return class weights for the rare-fraud supervised neural classifier."""
    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def mlp_positive_scores(model, X):
    """Return one positive-class score per transaction for a Keras MLP."""
    return model.predict(X, verbose=0).reshape(-1)


def train_mlp_experiments(
    X_train,
    y_train,
    X_val,
    y_val,
    configs=DEFAULT_MLP_CONFIGS,
    thresholds=DEFAULT_THRESHOLD_GRID,
    epochs=40,
    batch_size=1024,
    patience=5,
    random_state=DEFAULT_RANDOM_STATE,
):
    """Train MLP configs and tune supervised thresholds on validation data."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    class_weight = compute_balanced_class_weights(y_train)
    trained_models = {}
    histories = {}
    threshold_frames = []

    for config in configs:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_state)

        model = build_mlp_classifier(
            input_dim=X_train.shape[1],
            hidden_units=config["hidden_units"],
            dropout_rate=config["dropout_rate"],
            learning_rate=config["learning_rate"],
        )
        early_stop = EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
        )
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=1,
        )

        model_name = config["name"]
        val_scores = mlp_positive_scores(model, X_val)
        trained_models[model_name] = model
        histories[model_name] = history
        threshold_frames.append(
            tune_binary_thresholds(model_name, y_val, val_scores, thresholds)
        )

    validation_df = pd.concat(threshold_frames, ignore_index=True)
    return trained_models, histories, validation_df, class_weight


def reconstruction_error(model, X):
    """Compute row-wise mean squared reconstruction error."""
    reconstructions = model.predict(X, verbose=0)
    X_values = X.values if hasattr(X, "values") else np.asarray(X)
    return np.mean(np.square(X_values - reconstructions), axis=1)


def train_autoencoder_experiments(
    X_train_normal,
    X_val,
    y_val,
    configs=DEFAULT_AUTOENCODER_CONFIGS,
    threshold_percentiles=(90, 92, 95, 97, 99),
    epochs=50,
    batch_size=256,
    patience=5,
    random_state=DEFAULT_RANDOM_STATE,
):
    """Train autoencoder configs and evaluate thresholds on validation data."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    input_dim = X_train_normal.shape[1]
    trained_autoencoders = {}
    histories = {}
    validation_rows = []

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    for config in configs:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_state)

        model = build_autoencoder(
            input_dim=input_dim,
            bottleneck_size=config["bottleneck_size"],
            dropout_rate=config["dropout_rate"],
            learning_rate=config.get("learning_rate", 1e-3),
        )

        history = model.fit(
            X_train_normal,
            X_train_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            shuffle=True,
            verbose=1,
        )

        train_loss = reconstruction_error(model, X_train_normal)
        val_loss = reconstruction_error(model, X_val)

        model_name = config["name"]
        trained_autoencoders[model_name] = model
        histories[model_name] = history

        for percentile in threshold_percentiles:
            threshold = np.percentile(train_loss, percentile)
            row = evaluate_anomaly_scores(
                model_name,
                y_val,
                val_loss,
                threshold,
                percentile=percentile,
            )
            row["Bottleneck_Size"] = config["bottleneck_size"]
            row["Dropout_Rate"] = config["dropout_rate"]
            row["Learning_Rate"] = config.get("learning_rate", 1e-3)
            validation_rows.append(row)

    validation_df = pd.DataFrame(validation_rows)
    return trained_autoencoders, histories, validation_df


def save_autoencoder_training_curve(history, model_name, figures_dir):
    """Save the selected autoencoder training and validation loss curve."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_title(f"Autoencoder Training Curve - {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.legend()

    fig.tight_layout()
    output_path = figures_dir / f"{_safe_filename(model_name)}_training_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def best_validation_result_per_model(validation_df):
    """Return the validation-selected row for each model candidate."""
    ordered = validation_df.sort_values(
        ["F1", "PR_AUC", "Recall"],
        ascending=False,
    )
    return ordered.groupby("Model", sort=False, as_index=False).head(1)


def save_autoencoder_learning_rate_comparison(validation_df, figures_dir):
    """Compare validation metrics for a fixed-architecture autoencoder LR sweep."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    best_rows = best_validation_result_per_model(validation_df).sort_values(
        "Learning_Rate"
    )
    rate_labels = [f"{rate:.0e}" for rate in best_rows["Learning_Rate"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(best_rows))
    width = 0.36

    ax.bar(x - width / 2, best_rows["F1"], width, label="Validation F1")
    ax.bar(x + width / 2, best_rows["PR_AUC"], width, label="Validation PR-AUC")
    ax.set_title("Autoencoder Learning-Rate Sweep")
    ax.set_xlabel("Adam learning rate")
    ax.set_ylabel("Validation score")
    ax.set_xticks(x, rate_labels)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    output_path = figures_dir / "autoencoder_learning_rate_sweep_validation.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path, best_rows


def save_mlp_training_curve(history, figures_dir):
    """Save loss and PR-AUC curves for the selected MLP."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Training loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_title("MLP Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["pr_auc"], label="Training PR-AUC")
    axes[1].plot(history.history["val_pr_auc"], label="Validation PR-AUC")
    axes[1].set_title("MLP PR-AUC During Training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PR-AUC")
    axes[1].legend()

    fig.tight_layout()
    output_path = figures_dir / "mlp_selected_model_training_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_supervised_pipeline(processed_data_path, results_dir):
    """Train supervised models, tune thresholds on validation data, and test once."""
    processed_data_path = Path(processed_data_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = joblib.load(processed_data_path)
    candidate_models = get_supervised_candidate_models()
    trained_models = train_supervised_models(
        candidate_models,
        data["X_train"],
        data["y_train"],
    )

    default_df, validation_scores = evaluate_supervised_default_thresholds(
        trained_models,
        data["X_val"],
        data["y_val"],
    )
    threshold_df = tune_supervised_models(validation_scores, data["y_val"])
    best_model, best_name, best_threshold, best_row = select_supervised_model(
        trained_models,
        threshold_df,
    )

    test_scores = best_model.predict_proba(data["X_test"])[:, 1]
    test_df = pd.DataFrame(
        [
            evaluate_binary_scores(
                best_name,
                data["y_test"],
                test_scores,
                best_threshold,
            )
        ]
    )

    default_df.to_csv(results_dir / "baseline_validation_default_threshold.csv", index=False)
    threshold_df.to_csv(
        results_dir / "baseline_validation_threshold_tuning.csv",
        index=False,
    )
    test_df.to_csv(results_dir / "baseline_test_results.csv", index=False)

    joblib.dump(
        {
            "trained_models": trained_models,
            "best_model_name": best_name,
            "best_threshold": best_threshold,
            "best_validation_row": best_row.to_dict(),
        },
        results_dir / "baseline_models.pkl",
    )

    print("Supervised pipeline complete.")
    print(test_df.to_string(index=False))
    return test_df


def run_mlp_pipeline(processed_data_path, results_dir, figures_dir):
    """Train MLP candidates, tune thresholds on validation data, and test once."""
    processed_data_path = Path(processed_data_path)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = joblib.load(processed_data_path)
    trained_models, histories, validation_df, class_weight = train_mlp_experiments(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
    )

    best_row = select_best_result(validation_df)
    best_name = best_row["Model"]
    best_threshold = float(best_row["Threshold"])
    best_model = trained_models[best_name]
    best_history = histories[best_name]
    best_config = next(
        config for config in DEFAULT_MLP_CONFIGS if config["name"] == best_name
    )

    test_scores = mlp_positive_scores(best_model, data["X_test"])
    test_df = pd.DataFrame(
        [
            evaluate_binary_scores(
                best_name,
                data["y_test"],
                test_scores,
                best_threshold,
            )
        ]
    )

    validation_df.to_csv(
        results_dir / "mlp_validation_threshold_tuning.csv",
        index=False,
    )
    test_df.to_csv(results_dir / "mlp_test_results.csv", index=False)
    best_model.save(results_dir / "mlp_best_model.keras")
    training_curve_path = save_mlp_training_curve(best_history, figures_dir)

    joblib.dump(
        {
            "best_mlp_name": best_name,
            "best_mlp_threshold": best_threshold,
            "best_mlp_config": dict(best_config),
            "best_mlp_validation_row": best_row.to_dict(),
            "class_weight": class_weight,
            "training_curve_path": str(training_curve_path),
        },
        results_dir / "mlp_selection.pkl",
    )

    print("MLP pipeline complete.")
    print("Saved training curve:", training_curve_path)
    print(test_df.to_string(index=False))
    return test_df


def run_autoencoder_pipeline(processed_data_path, results_dir, figures_dir):
    """Train autoencoder experiments, tune thresholds on validation data, and test once."""
    processed_data_path = Path(processed_data_path)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = joblib.load(processed_data_path)
    X_train_normal = data["X_train"][data["y_train"] == 0]

    autoencoders, histories, validation_df = train_autoencoder_experiments(
        X_train_normal,
        data["X_val"],
        data["y_val"],
    )
    _, _, learning_rate_validation_df = train_autoencoder_experiments(
        X_train_normal,
        data["X_val"],
        data["y_val"],
        configs=DEFAULT_AUTOENCODER_LR_SWEEP_CONFIGS,
    )

    best_row = select_best_result(validation_df)
    best_name = best_row["Model"]
    best_threshold = float(best_row["Threshold"])
    best_model = autoencoders[best_name]
    test_loss = reconstruction_error(best_model, data["X_test"])

    test_df = pd.DataFrame(
        [
            evaluate_anomaly_scores(
                best_name,
                data["y_test"],
                test_loss,
                best_threshold,
                percentile=best_row["Percentile"],
            )
        ]
    )

    validation_df.to_csv(
        results_dir / "autoencoder_validation_threshold_tuning.csv",
        index=False,
    )
    test_df.to_csv(results_dir / "autoencoder_test_results.csv", index=False)
    best_model.save(results_dir / "autoencoder_best_model.keras")
    training_curve_path = save_autoencoder_training_curve(
        histories[best_name],
        best_name,
        figures_dir,
    )
    learning_rate_figure_path, learning_rate_summary_df = (
        save_autoencoder_learning_rate_comparison(
            learning_rate_validation_df,
            figures_dir,
        )
    )
    learning_rate_validation_df.to_csv(
        results_dir / "autoencoder_learning_rate_sweep_validation.csv",
        index=False,
    )
    learning_rate_summary_df.to_csv(
        results_dir / "autoencoder_learning_rate_sweep_best_validation.csv",
        index=False,
    )

    joblib.dump(
        {
            "best_autoencoder_name": best_name,
            "best_autoencoder_threshold": best_threshold,
            "best_autoencoder_validation_row": best_row.to_dict(),
            "training_curve_path": str(training_curve_path),
            "learning_rate_sweep_figure_path": str(learning_rate_figure_path),
        },
        results_dir / "autoencoder_selection.pkl",
    )

    print("Autoencoder pipeline complete.")
    print("Saved training curve:", training_curve_path)
    print("Saved learning-rate comparison:", learning_rate_figure_path)
    print(test_df.to_string(index=False))
    return test_df


def main():
    """Run model pipelines from the command line."""
    parser = argparse.ArgumentParser(description="Train ACML project models.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="supervised",
        choices=["supervised", "mlp", "autoencoder", "all"],
        help="Which model pipeline to run.",
    )
    parser.add_argument(
        "--data-path",
        default="outputs/results_corrected/creditcard_cleaned.pkl",
    )
    parser.add_argument("--results-dir", default="outputs/results_corrected")
    parser.add_argument("--figures-dir", default="outputs/figures")
    args = parser.parse_args()

    if args.mode in {"supervised", "all"}:
        run_supervised_pipeline(args.data_path, args.results_dir)

    if args.mode in {"mlp", "all"}:
        run_mlp_pipeline(args.data_path, args.results_dir, args.figures_dir)

    if args.mode in {"autoencoder", "all"}:
        run_autoencoder_pipeline(args.data_path, args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
