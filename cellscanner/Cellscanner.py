#!/usr/bin/env python
"""
==========================================
Overview
==========================================

CellScanner is a Python-based graphical user interface (GUI) application designed for analyzing flow cytometry data.
It specializes in predicting cell species within a coculture environment, facilitating the import of monoculture
and blank samples, training neural network models, and performing detailed analyses including gating for live/dead
cells and debris, as well as heterogeneity assessments.

==========================================
Key Features
==========================================

- **Data Importation**: Seamlessly import monoculture and blank flow cytometry samples for analysis.
- **Model Training**: Train neural network models on imported data to predict cell species in coculture samples.
- **Species Prediction**: Utilize trained models to predict species in new coculture samples with uncertainty assessments.
- **Gating Functionality**: Apply gating mechanisms to classify cells into live, inactive, or debris categories based on staining parameters.
- **Heterogeneity Analysis**: Perform heterogeneity assessments using both simple range-based methods and MiniBatchKMeans clustering.
- **Interactive Visualizations**: Generate and save interactive 3D scatter plots, pie charts, and bar charts to visualize predictions, gating results, and heterogeneity measures.

==========================================
Cellscanner.py
==========================================

Initializes the QApplication, instantiates the main GUI window, and executes the application loop to render the GUI.

Usage:
- Import Files: Allows users to import monoculture and blank files for analysis.
- Train Neural Network: Provides an interface to train a neural network model on the imported data.
- Predict Coculture: Allows users to select a coculture file, predict species within the sample, and optionally apply gating and heterogeneity analysis.
"""
import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,\
    QPushButton, QWidget, QLabel, QScrollArea

from scripts.helpers import get_app_dir
from scripts.GUIhelpers import button_style
from scripts.ImportFiles import  ImportFilePanel
from scripts.TrainingModel import TrainModelPanel
from scripts.Prediction import PredictionPanel

"""
Authors:
    - Ermis Ioannis Michail Delopoulos
    - Haris Zafeiropoulos

Date: 2024-2025
"""

class NeuralNetworkGUI(QMainWindow):
    """
    Main class of the CellScanner GUI.
    It builds a PyQT app with 3 main panels:
    - Importing files (:class:`scripts.ImportFiles.ImportFilePanel`)
    - Training (:class:`scripts.TrainingModel.TrainModelPanel`)
    - Prediction (:class:`scripts.Prediction.PredictionPanel`)
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CellScanner")
        # setGeometry(x, y, width, height) x, y stand for top-left corner
        self.setGeometry(100, 100, 950, 2100)

        # Initialize model-related attributes
        self.model = None
        self.scaler = None
        self.label_encoder = None

        # Central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Create a horizontal layout for the title and logo
        title_layout = QHBoxLayout()

        # Create and style the title
        self.title_label = QLabel("CellScanner", self)
        font = QFont("Arial", 30, QFont.Weight.Bold)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color: #2E8B57;")  # Dark sea green color
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Load the logo image
        logo_path = os.path.join(os.path.dirname(get_app_dir()), "logo.png")

        self.logo = QPixmap(logo_path)
        self.logo = self.logo.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.logo_label = QLabel(self)
        self.logo_label.setPixmap(self.logo)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Add the title and logo to the horizontal layout
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.logo_label)

        # Align the layout to the left
        title_layout.addStretch()

        # Add the title layout to the main layout
        self.layout.addLayout(title_layout)

        # NOTE: Initialize ImportFilePanel
        self.file_panel = ImportFilePanel(self)  # Instantiate ImportFilePanel

        # Initial Import Files Button with styling
        self.import_button = QPushButton("Import Data", self)
        self.import_button.setStyleSheet(
            button_style(
                font_size=16, padding=10, color="white",
                bck_col="#3b5998", bck_col_hov="#355089", bck_col_clicked="#2e477a",
                radius=8
            )
        )
        self.import_button.clicked.connect(self.toggle_file_panel)
        self.layout.addWidget(self.import_button)

        # Add the file import panel to the layout but hide it initially
        self.layout.addWidget(self.file_panel)
        self.file_panel.hide()  # Start with the panel hidden

        # NOTE: Training panel
        self.train_panel = TrainModelPanel(self.file_panel, self)  # Instantiate TrainModelPanel with the file pane
        self.train_button = QPushButton("Train Model", self)
        self.train_button.setStyleSheet(
            button_style(
                font_size=16, padding=10, color="white",
                bck_col="#3b5998", bck_col_hov="#355089", bck_col_clicked="#2e477a",
                radius=8
            )
        )
        self.train_button.clicked.connect(self.toggle_train_panel)
        # Add the train model panel to the layout but hide it initially
        self.layout.addWidget(self.train_button)
        self.layout.addWidget(self.train_panel)
        self.train_panel.hide()

        # NOTE: Prediction panel
        self.predict_panel = PredictionPanel(self.file_panel, self.train_panel, self)
        self.predict_button = QPushButton("Run Prediction", self)
        self.predict_button.setStyleSheet(
            button_style(
                font_size=16, padding=10, color="white",
                bck_col="#3b5998", bck_col_hov="#355089", bck_col_clicked="#2e477a",
                radius=8
            )
        )
        self.predict_button.clicked.connect(self.toggle_predict_panel)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.predict_panel)
        self.predict_panel.hide()

        # Create a QScrollArea and set it up
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # This makes the scroll area resize the content widget accordingly
        scroll_area.setWidget(central_widget)

        # Set the scroll area as the central widget of the main window
        self.setCentralWidget(scroll_area)

    def toggle_file_panel(self):
        """Shows or hides the data import panel based on user interaction."""
        if self.file_panel.isVisible():
            self.file_panel.hide()
        else:
            self.file_panel.show()

    def toggle_train_panel(self):
        """Shows or hides the model training panel."""
        if self.train_panel.isVisible():
            self.train_panel.hide()
        else:
            self.train_panel.show()

    def toggle_predict_panel(self):
        """Shows or hides the predict panel"""
        if self.predict_panel.isVisible():
            self.predict_panel.hide()
        else:
            self.predict_panel.show()


# Main application loop
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NeuralNetworkGUI()
    window.show()
    sys.exit(app.exec())
