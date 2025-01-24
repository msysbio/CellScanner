import os
import joblib

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from scipy.stats import entropy

def train_neural_network(TrainPanel=None, **kwargs):

    gui = False
    # if isinstance(TrainPanel, TrainModelPanel):
    if type(TrainPanel).__name__ == "TrainModelPanel":
        fold_count = int(TrainPanel.kfold_combo.currentText())
        epochs = int(TrainPanel.epochs_combo.currentText())
        batch_size = int(TrainPanel.batch_combo.currentText())
        patience = int(TrainPanel.patience_combo.currentText())
        X, y = TrainPanel.X, TrainPanel.y
        species_names = TrainPanel.le.classes_
        working_directory = TrainPanel.file_panel.working_directory
        gui = True

    else:
        fold_count = kwargs["fold_count"]
        epochs = kwargs["epochs"]
        batch_size = kwargs["batch_size"]
        patience = kwargs["patience"]
        X, y = kwargs["X"], kwargs["y"],
        species_names=kwargs["species_names"]
        working_directory=kwargs["working_directory"]

    if X is None or y is None:
        raise ValueError("No dataset loaded. Please run prepare_for_training first.")

    # Convert one-hot y back to integer if needed
    y_int = np.argmax(y, axis=1)
    model_dir = os.path.join(working_directory, "model")  # get_abs_path('model/statistics')

    # -------------- IF user chooses 0 folds --------------
    if fold_count == 0:

        # Just do a single hold-out approach (e.g., 80-20 split)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y_int
        )
        print("No cross-validation; using a single train/val split (80-20).")

        # Build a fresh model
        input_dim = X_train.shape[1]
        num_classes = y_train.shape[1]
        model = build_model(input_dim, num_classes)

        # Class weights (optional)
        y_train_int = np.argmax(y_train, axis=1)

        # Run training step
        model, val_loss, val_accuracy = train_wrapper(model, X_train, y_train, y_train_int,
            X_val, y_val, epochs, batch_size, patience
        )
        print(f"Single Split -> val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}")

        # Save the trained model
        model.save(os.path.join(model_dir, 'trained_model.keras'))

    # -------------- IF user chooses 5 or 10 folds --------------
    else:
        # Implement StratifiedKFold with that many folds
        skf = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)

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

            model, val_loss, val_accuracy = train_wrapper(model,
                        X_train, y_train, y_train_int,
                        X_val, y_val, epochs, batch_size, patience
            )
            fold_accuracies.append(val_accuracy)
            print(f"Fold {fold_idx} -> val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_fold = fold_idx
                best_model = model
            fold_idx += 1

        # Save best model
        best_model.save(os.path.join(model_dir, 'trained_model.keras'))

        # Re-run the split to get best fold's data for confusion matrix
        fold_idx = 1
        for train_idx, val_idx in skf.split(X, y_int):
            if fold_idx == best_fold:
                X_val_bf, y_val_bf = X[val_idx], y[val_idx]
                break
            fold_idx += 1

    # Save models and stats and calculate threshold
    trained_model = best_model if 'best_model' in locals() else model
    threshold = save_train_stats(
        trained_model,
        X_val_bf if 'X_val_bf' in locals() else X_val,
        y_val_bf if 'y_val_bf' in locals() else y_val,
        species_names,
        model_dir,
        best_accuracy if 'best_accuracy' in locals() else val_accuracy,
        best_fold if 'best_fold' in locals() else None,
        fold_count
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
        scaling_constant = TrainPanel.scaling_constant.value()
        working_directory = TrainPanel.file_panel.working_directory
        gui = True
    else:
        cleaned_data = kwargs["cleaned_data"]
        scaler = kwargs["scaler"]
        le = kwargs["le"]
        scaling_constant = kwargs["scaling_constant"]
        working_directory = kwargs["working_directory"]

    if cleaned_data is None:
        raise ValueError("No cleaned data available for training. Please process the data first.")

    # Make a copy of the cleaned data
    cleaned_data_copy = cleaned_data.copy()

    # 2. Separate features and labels
    X = cleaned_data_copy.drop('Species', axis=1)
    y_species = cleaned_data_copy['Species'].values

    # 3. arcsinh transform
    X_arcsinh = np.arcsinh(X / scaling_constant)

    # 4. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arcsinh)

    # (Skipping PCA, so X_whitened = X_scaled)
    X_whitened = X_scaled

    # Save scaler for future use/prediction
    model_dir = os.path.join(working_directory, "model")  # get_abs_path('model/statistics')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    # 5. Label encoding -> one-hot
    le = LabelEncoder()
    y_int = le.fit_transform(y_species)
    y_categorical = to_categorical(y_int)

    joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))

    if gui:
        # Store entire dataset
        TrainPanel.X = X_whitened
        TrainPanel.y = y_categorical
        TrainPanel.scaler = scaler
        TrainPanel.le = le
        print("Success: Data preparation done.")

        train_neural_network(TrainPanel)

    else:
        return X_whitened, y_categorical, scaler, le


def build_model(input_dim, num_classes):
    """
    Helper function to build a fresh model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_wrapper(model, X_train, y_train, y_train_int, X_val, y_val, epochs, batch_size, patience):
    """
    Helper function to train the model and return validation accuracy.
    """
    cw = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weight_dict = dict(enumerate(cw))

    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.01,
        patience=patience,
        mode='max',
        verbose=1
    )

    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=0
    )

    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    return model, val_loss, val_accuracy


def save_train_stats(model, X_val_bf, y_val_bf, species_names, model_dir, best_accuracy,
                     fold_count, best_fold=None):

    # Predict validation data
    conf_matrix_df, class_report_df, threshold = predict_validation(model, X_val_bf, y_val_bf, species_names)

    # Save stats
    stats_path = (os.path.join(model_dir, 'model_statistics_kfold.csv')
                  if fold_count is not None
                  else os.path.join(model_dir, 'model_statistics.csv')
    )
    with open(stats_path, 'w') as f:

        if best_fold is not None:
            f.write(f"Folds: {fold_count}\n")
            f.write(f"Best Fold: {best_fold}\n")

        f.write(f"Best Accuracy: {best_accuracy:.4f}")
        f.write("Threshold for Uncertainty: {:.4f}\n\n".format(threshold))

        f.write("Confusion Matrix:\n")
        conf_matrix_df.to_csv(f, header=True, index=True)
        f.write("\nClassification Report:\n")
        class_report_df.to_csv(f, header=True, index=True)

    if best_fold is not None:
        print(f"K-Fold training done. Best fold = {best_fold} with accuracy = {best_accuracy:.4f}. Stats saved.")
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
        y_true_classes, y_pred_classes,
        target_names=species_names,
        output_dict=True,
        zero_division=0
    )
    class_report_df = pd.DataFrame(class_report_dict).T

    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=species_names,
        columns=species_names
    )

    threshold = calculate_threshold(uncertainties, y_pred_classes, y_true_classes, species_names)

    return conf_matrix_df, class_report_df, threshold



def calculate_threshold(uncertainties, y_pred_classes, y_true_classes, species_names):

    threshold_report = {}
    max_threshold = np.max(uncertainties)

    for quantile in range(0, 100, 10):
        threshold = quantile / 100 * max_threshold
        try:
            # Filter out predictions with uncertainty below threshold
            valid_indices = np.where(uncertainties < threshold)[0]
            y_pred_classes_filtered = y_pred_classes[valid_indices]
            y_true_classes_filtered = y_true_classes[valid_indices]

            # Calculate classification report for the threshold under consideration
            class_report_dict = classification_report(
                y_true=y_true_classes_filtered,
                y_pred=y_pred_classes_filtered,
                target_names=species_names,
                output_dict=True,
                zero_division=0
            )
            threshold_report[threshold] = class_report_dict
        except:
            print(f"Threshold {threshold} does not return all {len(species_names)}.")
            threshold_report[threshold] = None
            pass

    best_accuracy = 0.0 ; best_threshold = 0.0
    for threshold, report in threshold_report.items():
        if report is not None:
            accuracy = report['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
    import json
    with open("all_thresholds_reports.json", 'w') as f:
        json.dump(threshold_report, f)

    return best_threshold

