"""
This script is used within the CellScanner application to predict species in coculture samples, apply gating for live/dead or debris
classification, and perform heterogeneity analysis.

Key Features:
- Predicts species in flow cytometry data using a trained neural network model.
- Applies gating to distinguish live, inactive, and debris states based on specified thresholds.
- Saves prediction and gating results, including visualizations in 3D scatter plots.
- Performs heterogeneity analysis using simple and MiniBatchKMeans clustering approaches.
- Generates and saves heterogeneity plots as HTML files.

Authors:
    - Ermis Ioannis Michail Delopoulos
    - Haris Zafeiropoulos

Date: 2024 - 2025
"""
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QGroupBox, QLabel, QMessageBox, QLineEdit, QCheckBox, QFileDialog, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

import os
import fcsparser
import numpy as np

from .helpers import button_style, time_based_dir
from .run_prediction import predict, merge_prediction_results
from .GUIhelpers import (
    AxisSelector, LiveDeadDebrisSelectors, GatingMixin, GatingCheckBox, GuiMessages,
    iterate_stains, load_fcs_file
)


class PredictionPanel(QWidget, GatingMixin, GatingCheckBox, LiveDeadDebrisSelectors):
    """
    The prediction panel of the CellScanner GUI enabling the user to provide files and parameters for predicting
    co-culture data.
    """
    def __init__(self, file_panel, train_panel, parent=None):

        super().__init__(parent)

        # Allow using ImportFilePanel and TrainModelPanel attributes directly
        self.file_panel = file_panel
        self.train_panel = train_panel

        # Init a QVBoxLayout as main layout of the panel
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
        self.axis_selection_label = QLabel(GuiMessages.AXIS_SELECTION, self)
        self.predict_panel_layout.addWidget(self.axis_selection_label)

        # X,Y,Z axis
        self.x_axis_selector = AxisSelector("X Axis:", self)
        self.y_axis_selector = AxisSelector("Y Axis:", self)
        self.z_axis_selector = AxisSelector("Z Axis:", self)

        self.predict_panel_layout.addWidget(self.x_axis_selector)
        self.predict_panel_layout.addWidget(self.y_axis_selector)
        self.predict_panel_layout.addWidget(self.z_axis_selector)

        self.axis_selectors = [
            self.x_axis_selector,
            self.y_axis_selector,
            self.z_axis_selector
        ]

        # Add a checkbox to apply uncertainty filtering
        self.uncertainty_filtering_checkbox = QCheckBox(GuiMessages.UNCERTAINTY_CHECKBOX, self)
        self.uncertainty_filtering_checkbox.stateChanged.connect(self.toggle_uncertainty_filterint_options)
        self.predict_panel_layout.addWidget(self.uncertainty_filtering_checkbox)

        # Scaling constant for uncertainty filtering
        self.uncertainty_threshold_layout = QHBoxLayout()
        self.uncertainty_threshold_label = QLabel("Threshold for uncertainty filtering:", self)
        self.uncertainty_threshold_layout.addWidget(self.uncertainty_threshold_label)

        self.uncertainty_threshold = QDoubleSpinBox(self)
        self.uncertainty_threshold.setToolTip(GuiMessages.UNCERTAINTY_TOOLTIP)
        self.uncertainty_threshold.setRange(-1.0, 10.0)
        self.uncertainty_threshold.setSingleStep(0.01)
        self.update_uncertainty_threshold()
        self.uncertainty_threshold_layout.addWidget(self.uncertainty_threshold)

        self.predict_panel_layout.addLayout(self.uncertainty_threshold_layout)

        # Add a checkbox to apply gating
        self.gating_checkbox()  # NOTE: from the GatingCheckBox mixin class, passed in the class definition

        # Fire basic stains
        self.basic_stains()  # NOTE: from the LiveDeadDebrisSelectors mixin class, passed in the class definition

        # Add user's labeled stain
        self.new_stain_button =  QPushButton(text="Add extra stain", parent=self)
        self.new_stain_button.setFixedSize(130, 20)
        self.new_stain_button.setStyleSheet(button_style(font_size=12, padding=5))
        self.new_stain_button.clicked.connect(self.build_stain_inputs)
        self.predict_panel_layout.addWidget(self.new_stain_button)

        # Hide gating and uncertainty filtering options initially
        self.toggle_gating_options()  # NOTE: from the GatingMixin mixin class, passed in the base classes of the PredictionPanel
        self.toggle_uncertainty_filterint_options()

        # Add gating layout to the predict one
        self.predict_panel_layout.addLayout(self.gating_layout)  # NOTE: (clarification) the gating_layout is there, thanks to the gating_checkbox mixin class

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
            self.stop_loading_cursor()

    def choose_coculture_file(self):
        select_coculture_message = ["Select Coculture File", "", "Flow Cytometry Files (*.fcs);;All Files (*)"]
        coculture_filepath, _ = QFileDialog.getOpenFileNames(self, *select_coculture_message)
        if coculture_filepath:

            try:
                # Load fcs files
                sample_to_df, sample_numeric_columns, numeric_columns = load_fcs_file(coculture_filepath)

                # Show files selected in the button
                self.choose_coculture_file_button.setText(",".join(sample_to_df.keys()))  # Display only the filename, not the full path

                # Check if all files share the same numeric column names
                all_same = all(value.equals(list(sample_numeric_columns.values())[0]) for value in sample_numeric_columns.values())
                if not all_same:
                    self.on_error(GuiMessages.COLUMN_NAMES_ERROR)

                # Populate the combo boxes with the numeric column names
                self.numeric_colums_set = set(numeric_columns)

                # Update all axis selectors
                for selector in self.axis_selectors:
                    selector.set_items(self.numeric_colums_set)

                # Update all stain selectors
                for selector in self.stain_selectors:
                    selector.set_items(self.numeric_colums_set)

                self.channels_on_stain_buttons()

                # Keep dictionary with sample names (key) and their corresponding data_df (value)
                self.sample_to_df = sample_to_df
            except:
                self.on_error("Something went off with your coculture files.")
        else:
            print("No coculture file selected.")
            self.choose_coculture_file_button.setText(select_coculture_message[0])

    def on_error(self, message):
        try:
            self.stop_loading_cursor()
            QMessageBox.critical(self, "Error", message)
        except Exception as e:
            print(f"Error displaying the message: {e}")
        finally:
            # Ensure that the thread is not running after error
            self.thread = None

    def start_loading_cursor(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)

    def stop_loading_cursor(self):
        QApplication.restoreOverrideCursor()

    def prediction_completed(self):
        self.stop_loading_cursor()
        QMessageBox.information(self, "Prediction Complete", f"Predictions have been saved in {self.predict_dir}.")
        self.thread = None

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

    def build_stain_inputs(self):
        """
        Function to support user-labeled stains
        A stain needs to have:
        - a channel
        - a sign
        - a threshold
        - a label
        CellScanner will set the label as True when the threshold holds.
        """
        stain_layout = QHBoxLayout()
        stain_description = QLabel("Staining cells:", self)
        stain_combo = QComboBox(self)
        stain_combo.setToolTip(GuiMessages.USER_STAIN_TOOLTIP)
        stain_relation = QComboBox(self)
        stain_relation.addItems(['>', '<'])
        stain_threshold = QDoubleSpinBox(self)  # QLineEdit(self)

        stain_label = QLineEdit(self)
        stain_label.setPlaceholderText("Enter label")

        stain_layout.addWidget(stain_description)
        stain_layout.addWidget(stain_combo)
        stain_layout.addWidget(stain_relation)
        stain_layout.addWidget(stain_threshold)
        stain_layout.addWidget(stain_label)

        container = QWidget(self)
        container.setLayout(stain_layout)
        self.gating_layout.addWidget(container)

        try:
            self.channels_on_stain_buttons()
        except:
            print("No coculture file yet.")
            pass
        stain_combo.addItem("Not applicable")

    def channels_on_stain_buttons(self):
        for i in range(self.gating_layout.count()):
            item = self.gating_layout.itemAt(i)  # Get the layout item
            widget = item.widget()  # Get the associated widget
            if widget:  # Ensure it's a QWidget
                # Access individual components within the stain container
                stain_layout = widget.layout()
                if stain_layout:
                    # Iterate through components of this layout
                    for j in range(stain_layout.count()):
                        component = stain_layout.itemAt(j).widget()
                        if isinstance(component, QComboBox):
                            if not any(component.itemText(i) == '>' for i in range(component.count())):
                                component.addItems(self.numeric_colums_set)


class WorkerPredict(QObject):
    """
    Worker class for running predict() for each co-culture file loaded, and if more than one, merge the findings
    """
    finished_signal = pyqtSignal()  # Define a signal for completion
    error_signal = pyqtSignal(str)

    def __init__(self, PredictPanel=None):
        super().__init__()
        self.PredictPanel = PredictPanel  # Store the QWidget instance

    def run_predict(self):
        """
        Main function to call the predict() function of CellScanner for each and every co-culture file provided
        """
        try:
            # Get dictioanry with user-labeled stains
            extra_stains = iterate_stains(self.PredictPanel)
            self.PredictPanel.extra_stains = extra_stains if len(extra_stains) > 0 else None
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
                predict(self.PredictPanel)

            # Merge predictions in case of multiple coculture files
            if multiple_cocultures:

                print("Merge prediction of all samples into a single file.")
                merge_prediction_results(self.PredictPanel.predict_dir, "prediction")

                print("Merge prediction and uncertainties single file.")
                merge_prediction_results(self.PredictPanel.predict_dir, "uncertainty")

            self.finished_signal.emit()  # Emit the finished signal when done

        except Exception as e:
            self.error_signal.emit(f"Error during prediction: {str(e)}")
            self.PredictPanel.thread.quit()

