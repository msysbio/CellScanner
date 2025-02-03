from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,\
    QGroupBox, QLabel, QMessageBox, QApplication, QSpinBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

from .helpers import button_style

from .apply_umap import process_files

"""
TrainingModel.py

This module is part of the CellScanner application, responsible for processing flow cytometry data, training
a neural network model, and evaluating its performance. The `TrainModelPanel` class provides a user interface
panel for selecting training parameters and initiating the training process.

Key Features:
- Allows users to select the number of random events to sample from each monoculture and blank file.
- Handles the preprocessing of `.fcs` files, including file parsing, data sampling, and scaling.
- Implements UMAP for dimensionality reduction and filtering of data based on nearest neighbors.
- Trains a neural network model using the processed data.
- Evaluates the trained model and saves performance metrics, including a confusion matrix and classification report.

Classes:
- TrainModelPanel: A QWidget subclass that provides the interface for training the neural network model.

Usage:
- The `TrainModelPanel` is integrated into the main application window and handles the entire model training pipeline.

Authors: Ermis Ioannis Michail Delopoulos, Haris Zafeiropoulos
Date: 22/08/2024
"""

class TrainModelPanel(QWidget):

    def __init__(self, file_panel, parent=None):

        super().__init__(parent)
        self.file_panel = file_panel
        self.layout = QVBoxLayout(self)

        #group box for "File Settings"
        self.file_settings_group = QGroupBox("File Settings", self)
        file_settings_layout = QVBoxLayout(self.file_settings_group)

        # Init cs uncertainty threshold to None
        self.cs_uncertainty_threshold = None

        # Random Events Selection using QComboBox
        event_layout = QHBoxLayout()
        self.event_label = QLabel("Select number of random events:", self)
        self.event_label.setToolTip(
            "Select the number of random events to sample from each monoculture and blank file."
        )

        self.event_combo = QComboBox(self)
        self.event_combo.addItems(
            [str(i) for i in range(1000, 100000, 5000)]
        )  # Adds 1000, 2000, ..., up to 95,000
        event_layout.addWidget(self.event_label)
        event_layout.addWidget(self.event_combo)

        # Message for gating parameters
        gating_layout = QHBoxLayout()
        message_for_gating = QLabel(
            "If you plan to apply line gating for live/dead cells and/or cells vs. debris during the prediction step,\n"
            "please enter the relevant parameters in the designated boxes within the Prediction panel.\n"
            "CellScanner will incorporate stainings for inactive and total cells to enhance model training.",
            self
        )
        gating_layout.addWidget(message_for_gating)

        # Add the event_layout into file_settings_layout, then add file_settings_group to the main layout.
        file_settings_layout.addLayout(event_layout)
        file_settings_layout.addLayout(gating_layout)
        self.layout.addWidget(self.file_settings_group)

        # UMAP Settings GroupBox
        self.umap_group = QGroupBox("UMAP Settings", self)
        umap_layout = QVBoxLayout(self.umap_group)

        #  UMAP n_neighbors
        umap_nneighbors_layout = QHBoxLayout()
        umap_nneighbors_label = QLabel("UMAP n_neighbors:", self)
        self.umap_nneighbors_combo = QComboBox(self)

        # Provide some typical neighbor-size choices for UMAP
        self.umap_nneighbors_combo.addItems(["5","10","15", "30", "50", "100", "200"])
        self.umap_nneighbors_combo.setCurrentText("50")  # default
        umap_nneighbors_layout.addWidget(umap_nneighbors_label)
        umap_nneighbors_layout.addWidget(self.umap_nneighbors_combo)
        umap_layout.addLayout(umap_nneighbors_layout)

        # UMAP min_dist
        umap_mindist_layout = QHBoxLayout()
        umap_mindist_label = QLabel("UMAP min_dist:", self)
        self.umap_mindist_combo = QComboBox(self)
        self.umap_mindist_combo.addItems(["0.0", "0.1","0.2", "0.3", "0.4", "0.5", "0.6","0.7","0.8","0.9"])
        self.umap_mindist_combo.setCurrentText("0.0")  # default
        umap_mindist_layout.addWidget(umap_mindist_label)
        umap_mindist_layout.addWidget(self.umap_mindist_combo)
        umap_layout.addLayout(umap_mindist_layout)

        # Add the UMAP group box to the main layout
        self.layout.addWidget(self.umap_group)

        # -----------------------------------------------------------

        # Nearest Neighbors Settings GroupBox
        self.nn_group = QGroupBox("Nearest Neighbors Threshold Settings (NN is 50)", self)
        nn_layout = QVBoxLayout(self.nn_group)

        # Non-blank threshold
        nn_nonblank_layout = QHBoxLayout()
        nn_nonblank_label = QLabel("Neighbor threshold (non-Blank):", self)
        self.nn_nonblank_combo = QComboBox(self)
        # Add the allowed threshold values
        self.nn_nonblank_combo.addItems(["10","15", "20","25", "30","35", "40"])
        # Set the default ( "25")
        self.nn_nonblank_combo.setCurrentText("25")

        nn_nonblank_layout.addWidget(nn_nonblank_label)
        nn_nonblank_layout.addWidget(self.nn_nonblank_combo)
        nn_layout.addLayout(nn_nonblank_layout)

        # Blank threshold
        nn_blank_layout = QHBoxLayout()
        nn_blank_label = QLabel("Neighbor threshold (Blank):", self)
        self.nn_blank_combo = QComboBox(self)
        # Add the allowed threshold values
        self.nn_blank_combo.addItems(["10","15","20","25", "30", "35", "40"])
        # Set the default ("20")
        self.nn_blank_combo.setCurrentText("20")

        nn_blank_layout.addWidget(nn_blank_label)
        nn_blank_layout.addWidget(self.nn_blank_combo)
        nn_layout.addLayout(nn_blank_layout)

        # SCALING CONSTANT
        scaling_constant_layout = QHBoxLayout()
        scaling_constant_label = QLabel("Scaling Constant:", self)
        scaling_constant_layout.addWidget(scaling_constant_label)

        self.scaling_constant = QSpinBox(self)
        self.scaling_constant.setRange(0, 1000)  # Set minimum and maximum values
        self.scaling_constant.setSingleStep(1)  # Set step size
        self.scaling_constant.setValue(150)  # Set default value
        scaling_constant_layout.addWidget(self.scaling_constant)

        nn_layout.addLayout(scaling_constant_layout)

        # Add the NN group box to the main layout
        self.layout.addWidget(self.nn_group)

        # Model specific options grooup
        self.model_settings_group = QGroupBox("Model Specific Settings", self)
        model_settings_layout = QVBoxLayout(self.model_settings_group)

        # K-Fold selection
        kfold_layout = QHBoxLayout()
        self.kfold_label = QLabel("Number of Folds:", self)
        kfold_layout.addWidget(self.kfold_label)

        self.kfold_combo = QComboBox(self)
        self.kfold_combo.addItems(["0", "5", "10"])
        kfold_layout.addWidget(self.kfold_combo)

        model_settings_layout.addLayout(kfold_layout)  #  Add to model_settings_layout

        # EPOCHS
        epochs_layout = QHBoxLayout()
        self.epochs_label = QLabel("Epochs:", self)
        epochs_layout.addWidget(self.epochs_label)

        self.epochs_combo = QComboBox(self)
        self.epochs_combo.addItems(["10", "20", "30", "50", "100","500"])
        self.epochs_combo.setCurrentText("50")
        epochs_layout.addWidget(self.epochs_combo)

        model_settings_layout.addLayout(epochs_layout)

        # BATCH SIZE
        batch_layout = QHBoxLayout()
        self.batch_label = QLabel("Batch Size:", self)
        batch_layout.addWidget(self.batch_label)

        self.batch_combo = QComboBox(self)
        self.batch_combo.addItems(["16", "32", "64", "128"])
        self.batch_combo.setCurrentText("32")
        batch_layout.addWidget(self.batch_combo)

        model_settings_layout.addLayout(batch_layout)

        # PATIENCE
        patience_layout = QHBoxLayout()
        self.patience_label = QLabel("EarlyStopping Patience:", self)
        patience_layout.addWidget(self.patience_label)

        self.patience_combo = QComboBox(self)
        self.patience_combo.addItems(["5", "10", "15", "20"])
        self.patience_combo.setCurrentText("10")
        patience_layout.addWidget(self.patience_combo)

        model_settings_layout.addLayout(patience_layout)

        # Add the QGroupBox to your main layout
        self.layout.addWidget(self.model_settings_group)

        # Process Files Button
        self.process_button = QPushButton("Train", self)
        self.process_button.setStyleSheet(button_style())

        self.process_button.clicked.connect(self.start_training_process)
        self.layout.addWidget(self.process_button)

        # Store the processed and filtered dataframe
        self.cleaned_data = None
        self.X = None
        self.y = None
        self.le = None
        self.scaler = None

        # Keep a reference to best model
        self.best_model = None


    def start_loading_cursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)

    def stop_loading_cursor(self):
        QApplication.restoreOverrideCursor()


    def start_training_process(self):
        # Load predict panel attributes to enable gating
        self.predict_panel = self.parent().parent().predict_panel

        try:
            #start the loading cursor
            self.start_loading_cursor()

            if not self.file_panel.species_files and not self.file_panel.blank_files:
                raise ValueError("No files selected. Please import files.")

            # Create a new thread for processing- without it the app froze while running Neural
            self.thread = QThread()
            self.worker = WorkerProcessFiles(parent_widget=self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run_process_files)

            # Apply UMAP & train neural network
            self.worker.finished_signal.connect(self.on_finished)
            self.worker.finished_signal.connect(self.thread.quit)

            # Ensure the thread finishes properly but does not exit the app
            self.thread.finished.connect(self.thread.deleteLater)

            # Thread-related tasks to perform and check if fine
            self.thread.start()

        except Exception as e:
           self.on_error(str(e))
           self.stop_loading_cursor()


    def on_finished(self):
        self.stop_loading_cursor()
        QMessageBox.information(self, "Success",
            f"Training process completed successfully. Suggested thresholds equals to {self.cs_uncertainty_threshold}"
        )
        self.thread = None

    def on_error(self, message):
        self.stop_loading_cursor()
        QMessageBox.critical(self, "Error", message)
        self.thread = None


class WorkerProcessFiles(QObject):

    finished_signal = pyqtSignal()  # Define a signal for completion

    def __init__(self, parent_widget=None):
        super().__init__()
        self.parent_widget = parent_widget  # Store the QWidget instance


    def run_process_files(self):
        self.parent_widget = process_files(self.parent_widget)
        self.finished_signal.emit()  # Emit the finished signal when done

