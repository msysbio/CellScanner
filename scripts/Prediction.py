#Prediction.py
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,\
    QGroupBox, QLabel, QMessageBox, QLineEdit, QCheckBox, QFileDialog, QDoubleSpinBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import numpy as np
import fcsparser
from .helpers import button_style, time_based_dir
from .run_prediction import predict, merge_prediction_results

"""
This script is used within the CellScanner application to predict species in coculture samples, apply gating for live/dead or debris
classification, and perform heterogeneity analysis.

Key Features:
- Predicts species in flow cytometry data using a trained neural network model.
- Applies gating to distinguish live, inactive, and debris states based on specified thresholds.
- Saves prediction and gating results, including visualizations in 3D scatter plots.
- Performs heterogeneity analysis using simple and MiniBatchKMeans clustering approaches.
- Generates and saves heterogeneity plots as HTML files.

Functions:
- predict_species(data_df, model, scaler): Applies transformations and predicts species using a trained model.
- apply_gating(data_df, stain1, stain1_relation, stain1_threshold, stain2=None, ...): Applies gating to classify cells.
- save_prediction_results(predicted_classes, data_df, ...): Saves the prediction results and generates a 3D scatter plot.
- save_gating_results(gated_data_df, output_dir, ...): Saves the gating results and generates a 3D scatter plot.
- hetero_simple(data): Calculates simple heterogeneity as the sum of mean ranges across all channels.
- hetero_mini_batch(data, type='av_diss'): Computes heterogeneity using MiniBatchKMeans clustering.
- save_heterogeneity_plots(hetero1, hetero2, output_dir): Generates and saves pie and bar charts for heterogeneity measures.

Authors: Ermis Ioannis Michail Delopoulos
Date: 2024 - 2025
"""

class PredictionPanel(QWidget):
    def __init__(self, file_panel, train_panel, parent=None):
        super().__init__(parent)
        self.file_panel = file_panel
        self.train_panel = train_panel

        self.layout = QVBoxLayout(self)

        # Create a group box for the prediction panel
        self.predict_panel = QGroupBox("Coculture Options for Prediction", self)
        self.predict_panel_layout = QVBoxLayout(self.predict_panel)
        self.layout.addWidget(self.predict_panel)

        # Choose coculture file
        self.choose_coculture_file_button = QPushButton("Select Coculture File", self)
        self.choose_coculture_file_button.setStyleSheet(button_style(font_size=12, padding=5))
        self.choose_coculture_file_button.clicked.connect(self.choose_coculture_file)
        self.predict_panel_layout.addWidget(self.choose_coculture_file_button)

        # Add a text label for selecting the x, y, z axes
        self.axis_selection_label = QLabel("Choose the Channels that will be used as x, y, z axis for the 3D plot:", self)
        self.predict_panel_layout.addWidget(self.axis_selection_label)

        # X Axis selection
        self.x_axis_layout = QHBoxLayout()
        self.x_axis_label = QLabel("X Axis:", self)
        self.x_axis_combo = QComboBox(self)
        self.x_axis_layout.addWidget(self.x_axis_label)
        self.x_axis_layout.addWidget(self.x_axis_combo)
        self.predict_panel_layout.addLayout(self.x_axis_layout)

        # Y Axis selection
        self.y_axis_layout = QHBoxLayout()
        self.y_axis_label = QLabel("Y Axis:", self)
        self.y_axis_combo = QComboBox(self)
        self.y_axis_layout.addWidget(self.y_axis_label)
        self.y_axis_layout.addWidget(self.y_axis_combo)
        self.predict_panel_layout.addLayout(self.y_axis_layout)

        # Z Axis selection
        self.z_axis_layout = QHBoxLayout()
        self.z_axis_label = QLabel("Z Axis:", self)
        self.z_axis_combo = QComboBox(self)
        self.z_axis_layout.addWidget(self.z_axis_label)
        self.z_axis_layout.addWidget(self.z_axis_combo)
        self.predict_panel_layout.addLayout(self.z_axis_layout)

        # Add a checkbox to apply gating
        self.gating_checkbox = QCheckBox("Apply line gating", self)
        self.gating_checkbox.stateChanged.connect(self.toggle_gating_options)
        self.predict_panel_layout.addWidget(self.gating_checkbox)

        # Stain 1 selection (for live/dead)
        self.stain1_layout = QHBoxLayout()
        self.stain1_label = QLabel("Staining inactive cells (e.g. PI):", self)
        self.stain1_combo = QComboBox(self)
        self.stain1_combo.setToolTip(
            "Select the channel that will be used for gating live/dead cells. "
            "All events where the threshold is met will be classified as dead."
        )
        self.stain1_relation = QComboBox(self)
        self.stain1_relation.addItems(['>', '<'])
        self.stain1_threshold = QLineEdit(self)
        self.stain1_threshold.setPlaceholderText("Enter threshold")
        self.stain1_layout.addWidget(self.stain1_label)
        self.stain1_layout.addWidget(self.stain1_combo)
        self.stain1_layout.addWidget(self.stain1_relation)
        self.stain1_layout.addWidget(self.stain1_threshold)
        self.predict_panel_layout.addLayout(self.stain1_layout)

        # Stain 2 selection (for debris, optional)
        self.stain2_layout = QHBoxLayout()
        self.stain2_label = QLabel("Staining all cells (e.g. SYBR/DAPI):", self)
        self.stain2_combo = QComboBox(self)
        self.stain2_combo.setToolTip(
            "Select the channel that will be used for gating all cells. "
            "All events where the threshold is met will be classified as cells. "
            "The rest of the events will be classified as debris."
        )
        self.stain2_relation = QComboBox(self)
        self.stain2_relation.addItems(['>', '<'])
        self.stain2_threshold = QLineEdit(self)
        self.stain2_threshold.setPlaceholderText("Enter threshold")
        self.stain2_layout.addWidget(self.stain2_label)
        self.stain2_layout.addWidget(self.stain2_combo)
        self.stain2_layout.addWidget(self.stain2_relation)
        self.stain2_layout.addWidget(self.stain2_threshold)
        self.predict_panel_layout.addLayout(self.stain2_layout)

        # Add a checkbox to apply uncertainty filtering
        self.uncertainty_filtering_checkbox = QCheckBox("Apply filtering on the predictions based on their uncertainty scores.", self)
        self.uncertainty_filtering_checkbox.stateChanged.connect(self.toggle_uncertainty_filterint_options)
        self.predict_panel_layout.addWidget(self.uncertainty_filtering_checkbox)

        # Scaling constant for uncertainty filtering
        self.uncertainty_threshold_layout = QHBoxLayout()
        self.uncertainty_threshold_label = QLabel("Threshold for uncertainty filtering:", self)
        self.uncertainty_threshold_layout.addWidget(self.uncertainty_threshold_label)

        self.uncertainty_threshold = QDoubleSpinBox(self)
        self.uncertainty_threshold.setToolTip(\
            "Set threshold for filtering out uncertain predictions. "
            "If you just trained a model, CellScanner computed already the threshold allowing the highest accuracy and set it as default. "
            "If you are loading a model, you can set the threshold manually, and if you are using a previously trained model, "
            "you can use its corresponding model_statistics file to remember the threshold suggested. "
            "To use the widely used threshold of 0.5 of the maximum entropy, set this value to -1.0 and CellScanner will apply this."
        )
        self.uncertainty_threshold.setRange(-1.0, 10.0)  # Set minimum and maximum values
        self.uncertainty_threshold.setSingleStep(0.01)  # Set step size
        self.update_uncertainty_threshold()
        self.uncertainty_threshold_layout.addWidget(self.uncertainty_threshold)

        self.predict_panel_layout.addLayout(self.uncertainty_threshold_layout)

        # Hide gating and uncertainty filtering options initially
        self.toggle_gating_options()
        self.toggle_uncertainty_filterint_options()

        # Run Prediction Button
        self.run_prediction_button = QPushButton("Predict", self)
        self.run_prediction_button.setStyleSheet(button_style(font_size=12, padding=5))
        self.run_prediction_button.clicked.connect(self.fire_predict)

        self.predict_panel_layout.addWidget(self.run_prediction_button)


    def fire_predict(self):
        try:
            self.start_loading_cursor()
            self.samples_number = len(self.sample_to_df)
            if self.samples_number == 0:
                raise ValueError("Coculture data have not been provided.")

            # Create a new thread for processing- without it the app froze while running Neural
            self.thread = QThread()
            self.worker = WorkerPredict(PredictPanel=self)
            self.worker.moveToThread(self.thread)
            self.worker.error_signal.connect(self.on_error)

            self.thread.started.connect(self.worker.run_predict)

            # Apply UMAP & train neural network
            self.worker.finished_signal.connect(self.prediction_completed)
            self.worker.finished_signal.connect(self.thread.quit)

            # Ensure the thread finishes properly but does not exit the app
            self.thread.finished.connect(self.thread.deleteLater)

            # Thread-related tasks to perform and check if fine
            self.thread.start()
        except Exception as e:
            self.on_error(str(e))


    def choose_coculture_file(self):
        select_coculture_message = ["Select Coculture File", "", "Flow Cytometry Files (*.fcs);;All Files (*)"]
        coculture_filepath, _ = QFileDialog.getOpenFileNames(self, *select_coculture_message)
        if coculture_filepath:

            try:
                sample_to_df = {}
                sample_numeric_columns = {}

                for coc in coculture_filepath:
                    _, data_df = fcsparser.parse(coc, reformat_meta=True)

                    # Drop the 'Time' column if it exists
                    if 'Time' in data_df.columns:
                        data_df = data_df.drop(columns=['Time'])
                        sample_file_basename = os.path.basename(coc)  # coc.split('/')[-1]
                        sample, _ = os.path.splitext(sample_file_basename)

                        # Ensure only numeric columns are used in combo boxes
                        numeric_columns = data_df.select_dtypes(include=[np.number]).columns
                        sample_numeric_columns[sample_file_basename] = numeric_columns
                        sample_to_df[sample] = data_df

                # Show files selected in the button
                self.choose_coculture_file_button.setText(",".join(sample_to_df.keys()))  # Display only the filename, not the full path

                # Check if all files share the same numeric column names
                all_same = all(value.equals(list(sample_numeric_columns.values())[0]) for value in sample_numeric_columns.values())
                if not all_same:
                    self.on_error("\
                        Column names on your coculture files differ. Please make sure you only include files sharing the same column names."
                    )
                numeric_colums_set = set(numeric_columns)
                # Populate the combo boxes with the numeric column names
                self.stain1_combo.addItems(numeric_colums_set)
                self.stain2_combo.addItems(numeric_colums_set)
                self.x_axis_combo.addItems(numeric_colums_set)
                self.y_axis_combo.addItems(numeric_colums_set)
                self.z_axis_combo.addItems(numeric_colums_set)
                # Keep dictionary with sample names (key) and their corresponding data_df (value)
                self.sample_to_df = sample_to_df
            except:
                self.on_error("Something went off with your coculture files.")
        else:
            print("No coculture file selected.")
            self.choose_coculture_file_button.setText(select_coculture_message[0])


    def on_error(self, message):
        self.stop_loading_cursor()
        QMessageBox.critical(self, "Error", message)
        self.thread = None

    def start_loading_cursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)


    def stop_loading_cursor(self):
        QApplication.restoreOverrideCursor()


    def prediction_completed(self):
        self.stop_loading_cursor()
        QMessageBox.information(self, "Prediction Complete", f"Predictions have been saved in {self.predict_dir}.")
        self.thread = None


    def toggle_gating_options(self):
        # Show or hide gating options based on checkbox state
        is_checked = self.gating_checkbox.isChecked()
        self.stain1_label.setVisible(is_checked)
        self.stain1_combo.setVisible(is_checked)
        self.stain1_relation.setVisible(is_checked)
        self.stain1_threshold.setVisible(is_checked)
        self.stain2_label.setVisible(is_checked)
        self.stain2_combo.setVisible(is_checked)
        self.stain2_relation.setVisible(is_checked)
        self.stain2_threshold.setVisible(is_checked)


    def toggle_uncertainty_filterint_options(self):
        is_checked = self.uncertainty_filtering_checkbox.isChecked()
        self.filter_out_uncertain = True
        self.uncertainty_threshold_label.setVisible(is_checked)
        self.uncertainty_threshold.setVisible(is_checked)
        self.update_uncertainty_threshold()


    def update_uncertainty_threshold(self):
        if self.train_panel.cs_uncertainty_threshold is not None:
            self.uncertainty_threshold.setValue(self.train_panel.cs_uncertainty_threshold)
        else:
            self.uncertainty_threshold.setValue(-1.0)  # Set default value


class WorkerPredict(QObject):

    finished_signal = pyqtSignal()  # Define a signal for completion
    error_signal = pyqtSignal(str)

    def __init__(self, PredictPanel=None):
        super().__init__()
        self.PredictPanel = PredictPanel  # Store the QWidget instance


    def run_predict(self):

        multiple_cocultures = True if self.PredictPanel.samples_number > 1 else False

        # Get output directory for the predictions
        self.PredictPanel.predict_dir = time_based_dir(
            prefix="Prediction",
            base_path=self.PredictPanel.file_panel.output_dir,
            multiple_cocultures=multiple_cocultures
        )
        os.makedirs(self.PredictPanel.predict_dir, exist_ok=True)

        # Loop over the coculture files and run predict()
        for sample, data_df in self.PredictPanel.sample_to_df.items():

            # Set data_df for the sample in process
            self.PredictPanel.data_df = data_df
            self.PredictPanel.sample = sample

            # Run predict() for a single sample
            try:
                predict(self.PredictPanel)
            except Exception as e:
                self.error_signal.emit(f"Error during prediction: {str(e)}")

        # Merge predictions in case of multiple coculture files
        if multiple_cocultures:

            print("Merge prediction of all samples into a single file.")
            merge_prediction_results(self.PredictPanel.predict_dir, "prediction")

            print("Merge prediction and uncertainties single file.")
            merge_prediction_results(self.PredictPanel.predict_dir, "uncertainty")

        self.finished_signal.emit()  # Emit the finished signal when done
