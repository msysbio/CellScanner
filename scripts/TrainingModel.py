from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,\
    QGroupBox, QLabel, QMessageBox, QApplication, QSpinBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

from .helpers import button_style
from .apply_umap import process_files
from .GUIhelpers import LabeledComboBox, LabeledSpinBox, LiveDeadDebrisSelectors, GatingMixin, GatingCheckBox

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

Authors:
 - Ermis Ioannis Michail Delopoulos
 - Haris Zafeiropoulos

Date: 2024-2025
"""

class TrainModelPanel(QWidget, LiveDeadDebrisSelectors, GatingMixin, GatingCheckBox):

    def __init__(self, file_panel, parent=None):
        """
        Training panel using the mixin classes for gating

        """
        super().__init__(parent)
        self.file_panel = file_panel
        self.layout = QVBoxLayout(self)

        # Group box for "File Settings"
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
        )
        event_layout.addWidget(self.event_label)
        event_layout.addWidget(self.event_combo)

        # Add the event_layout into file_settings_layout, then add file_settings_group to the main layout.
        file_settings_layout.addLayout(event_layout)
        self.layout.addWidget(self.file_settings_group)

        # -----------------------------------------------------------

        # UMAP Settings GroupBox
        self.umap_group = QGroupBox("UMAP Settings", self)
        umap_layout = QVBoxLayout(self.umap_group)

        #  UMAP n_neighbors
        self.umap_nneighbors_combo = LabeledComboBox(
            "UMAP n_neighbors:",
            ["5","10","15", "30", "50", "100", "200"],
            "50",
            self
        )

        # UMAP min_dist
        self.umap_mindist_combo = LabeledComboBox(
            "UMAP min_dist:",
            ["0.0", "0.1","0.2", "0.3", "0.4", "0.5", "0.6","0.7","0.8","0.9"],
            "0.0",
            self
        )

        # Add UMAP widgets
        umap_layout.addWidget(self.umap_nneighbors_combo)
        umap_layout.addWidget(self.umap_mindist_combo)

        # Add the UMAP group box to the main layout
        self.layout.addWidget(self.umap_group)

        # -----------------------------------------------------------

        # Nearest Neighbors Settings GroupBox
        self.nn_group = QGroupBox("Nearest Neighbors Threshold Settings (NN is 50)", self)
        nn_layout = QVBoxLayout(self.nn_group)

        # Non-blank threshold
        self.nn_nonblank_combo = LabeledComboBox(
            "Neighbor threshold (non-Blank):",
            ["10","15", "20","25", "30","35", "40"],
            "25",
            self
        )
        # Blank threshold
        self.nn_blank_combo = LabeledComboBox(
            "Neighbor threshold (Blank):",
            ["10","15","20","25", "30", "35", "40"],
            "20",
            self
        )
        # SCALING CONSTANT
        self.scaling_constant = LabeledSpinBox(
            "Scaling Constant:",
            min_value=0, max_value=1000, step=1, default_value=150
        )
        nn_layout.addWidget(self.scaling_constant)
        nn_layout.addWidget(self.nn_blank_combo)
        nn_layout.addWidget(self.nn_nonblank_combo)

        # Add the NN group box to the main layout
        self.layout.addWidget(self.nn_group)

        # -----------------------------------------------------------

        # Model specific options grooup
        self.model_settings_group = QGroupBox("Model Specific Settings", self)
        model_settings_layout = QVBoxLayout(self.model_settings_group)

        # K-Fold SELECTION
        self.kfold_combo = LabeledComboBox("Number of Folds:", ["0", "5", "10"], parent=self)

        # EPOCHS
        self.epochs_combo = LabeledComboBox("Epochs:", ["10", "20", "30", "50", "100","500"], "50", self)

        # BATCH SIZE
        self.batch_combo = LabeledComboBox("Batch Size:", ["16", "32", "64", "128"], "32", self)

        # PATIENCE
        self.patience_combo = LabeledComboBox("EarlyStopping Patience:", ["5", "10", "15", "20"], "10", self)

        # Add the widgets to the group box layout
        model_settings_layout.addWidget(self.epochs_combo)
        model_settings_layout.addWidget(self.kfold_combo)
        model_settings_layout.addWidget(self.batch_combo)
        model_settings_layout.addWidget(self.patience_combo)

        # Add the QGroupBox to your main layout
        self.layout.addWidget(self.model_settings_group)

        # -----------------------------------------------------------

        # Gating option at the training step
        self.train_gating = QGroupBox("Line gating", self)
        self.train_gating_layout = QVBoxLayout(self.train_gating)

        # Add a checkbox to apply gating
        self.gating_checkbox()  # NOTE: from the GatingCheckBox mixin class, passed in the class definition

        # Fire basic stains
        self.basic_stains()  # NOTE: from the LiveDeadDebrisSelectors mixin class, passed in the class definition

        # Initially hide / show after click on gating checkbox
        self.toggle_gating_options()

        self.layout.addWidget(self.train_gating)

        # -----------------------------------------------------------

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
        """
        Establihes a thread and a worker to execute the run_process_files().
        The signals of the worker allows not to exit the app in case of error
        and to return a success message when complete.
        """
        try:
            #start the loading cursor
            self.start_loading_cursor()

            if not self.file_panel.species_files and not self.file_panel.blank_files:
                raise ValueError("No files selected. Please import files.")

            # Create a new thread for processing- without it the app froze while running Neural
            self.thread = QThread()
            self.worker = WorkerProcessFiles(TrainModelPanel=self)
            self.worker.moveToThread(self.thread)
            self.worker.error_signal.connect(self.on_error)
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


    def on_finished(self):
        self.stop_loading_cursor()
        QMessageBox.information(self, "Success",
            f"Training process completed successfully. Suggested thresholds equals to {self.cs_uncertainty_threshold}"
        )
        self.thread = None

    def on_error(self, message):
        try:
            self.stop_loading_cursor()
            QMessageBox.critical(self, "Error", message)
        except Exception as e:
            print(f"Error displaying the message: {e}")
        finally:
            self.thread = None

# A worker class allows tasks without blocking the main UI thread,
class WorkerProcessFiles(QObject):
    """
    Worker class for processing files in a separate thread.

    This worker is responsible for running `process_files()` without freezing the main UI.
    It emits signals to indicate success or failure, allowing the main UI to handle errors properly.
    """
    finished_signal = pyqtSignal()  # Define a signal for completion
    error_signal = pyqtSignal(str)

    def __init__(self, TrainModelPanel=None):
        super().__init__()
        self.TrainModelPanel = TrainModelPanel  # Store the QWidget instance


    def run_process_files(self):
        try:
            self.TrainModelPanel = process_files(self.TrainModelPanel)
            self.finished_signal.emit()  # Emit the finished signal when done
        except Exception as e:
            print("fuck this")
            self.error_signal.emit(f"Error during prediction: {str(e)}")
            self.TrainModelPanel.thread.quit()

