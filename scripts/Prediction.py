#Prediction.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,\
    QGroupBox, QLabel, QMessageBox, QLineEdit, QCheckBox, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import numpy as np
import fcsparser
from .helpers import button_style
from .run_prediction import predict

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
        self.gating_checkbox = QCheckBox("Apply Gating for Live/Dead or Debris", self)
        self.gating_checkbox.stateChanged.connect(self.toggle_gating_options)
        self.predict_panel_layout.addWidget(self.gating_checkbox)

        # Stain 1 selection (for live/dead)
        self.stain1_layout = QHBoxLayout()
        self.stain1_label = QLabel("Stain1 (For Live/Dead Cells):", self)
        self.stain1_combo = QComboBox(self)
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
        self.stain2_label = QLabel("Stain2 (For Debris)(Optional):", self)
        self.stain2_combo = QComboBox(self)
        self.stain2_relation = QComboBox(self)
        self.stain2_relation.addItems(['>', '<'])
        self.stain2_threshold = QLineEdit(self)
        self.stain2_threshold.setPlaceholderText("Enter threshold")
        self.stain2_layout.addWidget(self.stain2_label)
        self.stain2_layout.addWidget(self.stain2_combo)
        self.stain2_layout.addWidget(self.stain2_relation)
        self.stain2_layout.addWidget(self.stain2_threshold)
        self.predict_panel_layout.addLayout(self.stain2_layout)

        # Hide gating options initially
        self.toggle_gating_options()

        # Run Prediction Button
        self.run_prediction_button = QPushButton("Predict", self)
        self.run_prediction_button.setStyleSheet(button_style(font_size=12, padding=5))
        self.run_prediction_button.clicked.connect(self.fire_predict)


        self.predict_panel_layout.addWidget(self.run_prediction_button)


    def fire_predict(self):

        if self.data_df is None or self.data_df.empty:
            raise ValueError("Coculture data have not been provided.")

        # Create a new thread for processing- without it the app froze while running Neural
        self.thread = QThread()
        self.worker = WorkerPredict(parent_widget=self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run_predict)

        # Apply UMAP & train neural network
        self.worker.finished_signal.connect(self.prediction_completed)
        self.worker.finished_signal.connect(self.thread.quit)

        # Ensure the thread finishes properly but does not exit the app
        self.thread.finished.connect(self.thread.deleteLater)

        # Thread-related tasks to perform and check if fine
        self.thread.start()


    def prediction_completed(self):
        QMessageBox.information(self, "Prediction Complete", f"Predictions have been saved in {self.results_dir}.")

    def choose_coculture_file(self):
        select_coculture_message = ["Select Coculture File", "", "Flow Cytometry Files (*.fcs);;All Files (*)"]
        coculture_filepath, _ = QFileDialog.getOpenFileName(self, *select_coculture_message)
        if coculture_filepath:
            _, data_df = fcsparser.parse(coculture_filepath, reformat_meta=True)
            self.choose_coculture_file_button.setText(coculture_filepath.split('/')[-1])  # Display only the filename, not the full path

            # Drop the 'Time' column if it exists
            if 'Time' in data_df.columns:
                self.data_df = data_df.drop(columns=['Time'])

            # Ensure only numeric columns are used in combo boxes
            numeric_columns = data_df.select_dtypes(include=[np.number]).columns

            # Populate the combo boxes with the numeric column names
            self.stain1_combo.addItems(numeric_columns)
            self.stain2_combo.addItems(numeric_columns)
            self.x_axis_combo.addItems(numeric_columns)
            self.y_axis_combo.addItems(numeric_columns)
            self.z_axis_combo.addItems(numeric_columns)

            print("Coculture file loaded and 'Time' column dropped.")

        else:
            print("No coculture file selected.")
            # coculture_filepath, _ = QFileDialog.getOpenFileName(self, *select_coculture_message)
            self.choose_coculture_file_button.setText(select_coculture_message[0])

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

class WorkerPredict(QObject):

    finished_signal = pyqtSignal()  # Define a signal for completion

    def __init__(self, parent_widget=None):
        super().__init__()
        self.parent_widget = parent_widget  # Store the QWidget instance


    def run_predict(self):
        self.parent_widget = predict(self.parent_widget)
        self.finished_signal.emit()  # Emit the finished signal when done

