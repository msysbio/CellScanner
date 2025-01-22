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
        print(f"Single Split -> val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}")

        # Save the trained model
        # model_dir = os.path.join(working_directory, "model")  # get_abs_path('model/statistics')
        model.save(os.path.join(model_dir, 'trained_model.keras'))

        #  confusion matrix, classification report
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        print("===========")
        print(y_pred)
        print("===========")
        print(y_pred_classes)
        print("===========")

        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        class_report_dict = classification_report(
            y_true_classes, y_pred_classes,
            target_names=species_names,
            output_dict=True
        )
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=species_names,
            columns=species_names
        )
        class_report_df = pd.DataFrame(class_report_dict).T

        # Save stats
        stats_path = os.path.join(model_dir, 'model_statistics.csv')
        with open(stats_path, 'w') as f:
            f.write(f"Single-split approach\n")
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            conf_matrix_df.to_csv(f, header=True, index=True)
            f.write("\nClassification Report:\n")
            class_report_df.to_csv(f, header=True, index=True)

        print("Done training with single split (no cross-validation).")

        if gui:
            TrainPanel.model = model

        else:
            return model

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
            cw = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_int),
                y=y_train_int
            )
            class_weight_dict = dict(enumerate(cw))

            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                min_delta=0.01,
                patience=patience,
                mode='max',
                verbose=1
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                class_weight=class_weight_dict,
                verbose=0
            )

            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            fold_accuracies.append(val_accuracy)
            print(f"Fold {fold_idx} -> val_accuracy={val_accuracy:.4f}, val_loss={val_loss:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_fold = fold_idx
                best_model = model
            fold_idx += 1

        # Summaries
        print("\nCross-validation complete.")
        print("Fold Accuracies:", fold_accuracies)
        print(f"Best fold = {best_fold} with accuracy = {best_accuracy:.4f}")

        # Save best model
        # model_dir = get_abs_path('model/statistics')
        best_model.save(os.path.join(model_dir, 'trained_model.keras'))

        # Re-run the split to get best fold's data for confusion matrix
        fold_idx = 1
        for train_idx, val_idx in skf.split(X, y_int):
            if fold_idx == best_fold:
                X_val_bf, y_val_bf = X[val_idx], y[val_idx]
                break
            fold_idx += 1

        y_pred = best_model.predict(X_val_bf)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val_bf, axis=1)

        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
        class_report_dict = classification_report(
            y_true_classes, y_pred_classes,
            target_names=species_names,
            output_dict=True
        )
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=species_names,
            columns=species_names
        )
        class_report_df = pd.DataFrame(class_report_dict).T

        # Save stats
        stats_path = os.path.join(model_dir, 'model_statistics_kfold.csv')
        with open(stats_path, 'w') as f:
            f.write(f"Folds: {fold_count}\n")
            f.write(f"Best Fold: {best_fold}\n")
            f.write(f"Best Fold Accuracy: {best_accuracy:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            conf_matrix_df.to_csv(f, header=True, index=True)
            f.write("\nClassification Report:\n")
            class_report_df.to_csv(f, header=True, index=True)

        print("K-Fold training done. Best model saved, stats saved.")

        if gui:
            TrainPanel.model = best_model

        else:
            return best_model


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
