import math
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from .helpers import save_run_parameters


def train_neural_network(TrainPanel=None, **kwargs):

    gui = False
    # if isinstance(TrainPanel, TrainModelPanel):
    if type(TrainPanel).__name__ == "TrainModelPanel":
        fold_count = int(TrainPanel.kfold_combo.combo.currentText())
        epochs = int(TrainPanel.epochs_combo.combo.currentText())
        batch_size = int(TrainPanel.batch_combo.combo.currentText())
        patience = int(TrainPanel.patience_combo.combo.currentText())
        seed = TrainPanel.seed.spin_box.value()
        X, y = TrainPanel.X, TrainPanel.y
        species_names = TrainPanel.le.classes_
        working_directory = TrainPanel.file_panel.working_directory
        gui = True

    else:
        fold_count = kwargs["fold_count"]
        epochs = kwargs["epochs"]
        batch_size = kwargs["batch_size"]
        patience = kwargs["patience"]
        seed = kwargs.get("seed")
        X, y = (
            kwargs["X"],
            kwargs["y"],
        )
        species_names = kwargs["species_names"]
        working_directory = kwargs["working_directory"]

    if X is None or y is None:
        raise ValueError("No dataset loaded. Please run prepare_for_training first.")

    # Seed Python, NumPy and TensorFlow (weight initialisation, dropout, shuffling) for reproducible runs
    if seed is not None:
        keras.utils.set_random_seed(seed)

    # Convert one-hot y back to integer if needed
    y_int = np.argmax(y, axis=1)
    model_dir = os.path.join(
        working_directory, "model"
    )  # get_abs_path('model/statistics')

    # -------------- IF user chooses 0 folds --------------
    if fold_count == 0:
        # Just do a single hold-out approach (e.g., 80-20 split)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y_int
        )
        print("No cross-validation; using a single train/val split (80-20).")

        # Build a fresh model
        input_dim = X_train.shape[1]
        num_classes = y_train.shape[1]
        model = build_model(input_dim, num_classes)

        # Class weights (optional)
        y_train_int = np.argmax(y_train, axis=1)

        # Run training step
        model, val_loss, val_accuracy = train_wrapper(
            model,
            X_train,
            y_train,
            y_train_int,
            X_val,
            y_val,
            epochs,
            batch_size,
            patience,
        )
        print(
            f"Single Split -> val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}"
        )

        # Save the trained model
        model.save(os.path.join(model_dir, "trained_model.keras"))
        trained_model, X_eval, y_eval = model, X_val, y_val
        best_accuracy, best_fold = val_accuracy, None

    # -------------- IF user chooses 5 or 10 folds --------------
    else:
        # Implement StratifiedKFold with that many folds
        skf = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)

        best_accuracy = 0.0
        best_fold = -1
        best_model = None
        fold_accuracies = []
        fold_idx = 1
        for train_idx, val_idx in skf.split(X, y_int):
            print(f"\n--- Fold {fold_idx}/{fold_count} ---")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Build a fresh model
            input_dim = X_train.shape[1]
            num_classes = y_train.shape[1]
            model = build_model(input_dim, num_classes)

            y_train_int = np.argmax(y_train, axis=1)

            model, val_loss, val_accuracy = train_wrapper(
                model,
                X_train,
                y_train,
                y_train_int,
                X_val,
                y_val,
                epochs,
                batch_size,
                patience,
            )
            fold_accuracies.append(val_accuracy)
            print(
                f"Fold {fold_idx} -> val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}"
            )

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_fold = fold_idx
                best_model = model
            fold_idx += 1

        # Save best model
        best_model.save(os.path.join(model_dir, "trained_model.keras"))

        # Re-run the split to get best fold's data for confusion matrix
        fold_idx = 1
        for train_idx, val_idx in skf.split(X, y_int):
            if fold_idx == best_fold:
                X_eval, y_eval = X[val_idx], y[val_idx]
                break
            fold_idx += 1
        trained_model = best_model

    # Save models and stats and calculate threshold
    threshold = save_train_stats(
        trained_model,
        X_eval,
        y_eval,
        species_names,
        model_dir,
        best_accuracy=best_accuracy,
        fold_count=fold_count,
        best_fold=best_fold,
    )
    save_run_parameters(
        os.path.join(model_dir, "training_parameters.yml"),
        {
            "folds": fold_count,
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping_patience": patience,
            "seed": seed,
            "best_accuracy": best_accuracy,
            "best_fold": best_fold,
            "suggested_uncertainty_threshold": threshold,
        },
    )
    # Return the best model
    if gui:
        TrainPanel.model = trained_model
        TrainPanel.cs_uncertainty_threshold = threshold
    else:
        return trained_model, threshold


def prepare_for_training(TrainPanel=None, **kwargs):

    gui = False
    if type(TrainPanel).__name__ == "TrainModelPanel":
        cleaned_data = TrainPanel.cleaned_data
        scaler = TrainPanel.scaler
        le = TrainPanel.le
        scaling_constant = TrainPanel.scaling_constant.spin_box.value()
        working_directory = TrainPanel.file_panel.working_directory
        gui = True
    else:
        cleaned_data = kwargs["cleaned_data"]
        scaler = kwargs["scaler"]
        le = kwargs["le"]
        scaling_constant = kwargs["scaling_constant"]
        working_directory = kwargs["working_directory"]

    if cleaned_data is None:
        raise ValueError(
            "No cleaned data available for training. Please process the data first."
        )

    # Make a copy of the cleaned data
    cleaned_data_copy = cleaned_data.copy()

    # 2. Separate features and labels
    X = cleaned_data_copy.drop("Species", axis=1)
    y_species = cleaned_data_copy["Species"].values

    # 3. arcsinh transform
    X_arcsinh = np.arcsinh(X / scaling_constant)

    # 4. Scaling
    # The standard score of a sample `x` :
    #       z = (x - u) / s
    # where `u` is the mean of the training samples and `s` is the standard deviation of the training samples.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arcsinh)
    # NOTE: Skipping PCA, so X_whitened in previous implementations, is now replaced by X_scaled
    # X_whitened = X_scaled

    # Save scaler for future use/prediction
    model_dir = os.path.join(
        working_directory, "model"
    )  # get_abs_path('model/statistics')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    save_run_parameters(
        os.path.join(model_dir, "training_parameters.yml"),
        {"scaling_constant": scaling_constant, "channels": list(X.columns)},
    )

    # 5. Label encoding -> one-hot
    le = LabelEncoder()
    y_int = le.fit_transform(y_species)
    y_categorical = to_categorical(y_int)

    joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

    if gui:
        # Store entire dataset
        TrainPanel.X = X_scaled
        TrainPanel.y = y_categorical
        TrainPanel.scaler = scaler
        TrainPanel.le = le
        print("Success: Data preparation done.")

        # In GUI, call the main function for training the Neural Network.
        train_neural_network(TrainPanel)

    else:
        return X_scaled, y_categorical, scaler, le


def build_model(input_dim, num_classes):
    """
    Helper function to build a fresh model.
    """
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_wrapper(
    model, X_train, y_train, y_train_int, X_val, y_val, epochs, batch_size, patience
):
    """
    Helper function to train the model and return validation accuracy.
    """
    cw = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_int), y=y_train_int
    )
    class_weight_dict = dict(enumerate(cw))

    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor="val_accuracy", min_delta=0.01, patience=patience, mode="max", verbose=1
    )

    # Train
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=0,
    )

    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    return model, val_loss, val_accuracy


def save_train_stats(
    model,
    X_val_bf,
    y_val_bf,
    species_names,
    model_dir,
    best_accuracy,
    fold_count,
    best_fold=None,
):

    # Predict validation data
    conf_matrix_df, class_report_df, threshold, threshold_curve = predict_validation(
        model, X_val_bf, y_val_bf, species_names
    )

    # Save stats
    missing = [name for name in species_names if name not in class_report_df.index]
    if missing:
        raise ValueError(f"Species missing from the classification report: {missing}")

    stats_path = (
        os.path.join(model_dir, "model_statistics_kfold.csv")
        if fold_count
        else os.path.join(model_dir, "model_statistics.csv")
    )
    threshold_curve.to_csv(
        os.path.join(model_dir, "uncertainty_threshold_curve.csv"), index=False
    )
    with open(stats_path, "w") as f:
        if best_fold is not None:
            f.write(f"Folds: {fold_count}\n")
            f.write(f"Best Fold: {best_fold}\n")

        f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Threshold for Uncertainty: {threshold:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        conf_matrix_df.to_csv(f, header=True, index=True)
        f.write("\nClassification Report:\n")
        class_report_df.to_csv(f, header=True, index=True)

    if best_fold is not None:
        print(
            f"K-Fold training done. Best fold = {best_fold} with accuracy = {best_accuracy:.4f}. Stats saved."
        )
    else:
        print("Done training with single split (no cross-validation).")

    return threshold


def predict_validation(model, X_val_bf, y_val_bf, species_names):
    """
    Predicts the species of the test data.
    """
    # Run prediction
    y_pred = model.predict(X_val_bf)

    uncertainties = entropy(y_pred, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val_bf, axis=1)

    class_report_dict = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=species_names,
        output_dict=True,
        zero_division=0,
    )
    class_report_df = pd.DataFrame(class_report_dict).T

    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=species_names, columns=species_names
    )

    threshold, threshold_curve = calculate_threshold(
        uncertainties, y_pred_classes, y_true_classes, species_names
    )

    return conf_matrix_df, class_report_df, threshold, threshold_curve


def calculate_threshold(
    uncertainties, y_pred_classes, y_true_classes, species_names, min_coverage=0.8
):
    """
    Returns the entropy threshold with the best accuracy among those keeping at least
    ``min_coverage`` of the validation events, along with the full accuracy-vs-coverage curve.

    Entropy is in nats (``scipy.stats.entropy`` default), so the maximum is ln(n_classes).
    Events with entropy above the threshold are the ones :func:`predict` marks as ``Unknown``.
    Without the coverage constraint, accuracy rises as the threshold tightens and the search
    would always return the strictest cut-off.
    """
    max_threshold = math.log(len(species_names))

    rows = []
    for quantile in range(5, 101, 5):
        threshold = quantile / 100 * max_threshold
        keep = uncertainties <= threshold
        accuracy = (
            (y_pred_classes[keep] == y_true_classes[keep]).mean()
            if keep.any()
            else np.nan
        )
        rows.append((threshold, keep.mean(), accuracy))
    curve = pd.DataFrame(rows, columns=["threshold", "coverage", "accuracy"])

    candidates = curve[curve["coverage"] >= min_coverage].dropna()
    best = (
        candidates.loc[candidates["accuracy"].idxmax()]
        if not candidates.empty
        else curve.iloc[-1]
    )
    return float(best["threshold"]), curve
