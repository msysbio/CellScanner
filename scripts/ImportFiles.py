# ImportFiles.py
import os
import shutil
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, \
    QFileDialog, QInputDialog, QMessageBox
import fcsparser

from .helpers import get_app_dir, get_abs_path, time_based_dir, button_style

"""
ImportFiles.py

This module is part of the CellScanner application.
The `ImportFilePanel` class provides a user interface panel for selecting and managing flow cytometry files
for monoculture species 1, monoculture species 2, and blank samples. The selected files are copied to a
working directory, where they can be processed by other components of the application.

Key Features:
- Allows users to select `.fcs` files for species 1, species 2, and blank samples.
- Files are copied to a working directory to ensure the original files remain unchanged.
- Displays metadata and the first few rows of data for each selected file.
- Handles file parsing using the `fcsparser` library.
- Supports cleanup of the working directory to remove copied files.

Classes:
- ImportFilePanel: A QWidget subclass that provides the interface for selecting files and managing the working directory.


Author: Ermis Ioannis Michail Delopoulos
Date: 22/08/2024

"""


class ImportFilePanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # Working directory for copies
        self.working_directory = time_based_dir(base_path=get_app_dir(), prefix='working_files')
        os.makedirs(self.working_directory, exist_ok=True)  # Ensure the directory exists

        # List to store species-related file paths and names
        self.species_widgets = []
        self.species_files = {}  # A dictionary to store species and their corresponding files

        # Blank Files Selection
        blank_layout = QHBoxLayout()
        self.blank_button = QPushButton("Select Blank Files (.fcs)", self)
        self.blank_button.setStyleSheet(button_style())
        self.blank_button.clicked.connect(self.select_blank_files)
        blank_layout.addWidget(self.blank_button)
        self.layout.addLayout(blank_layout)

        # storing blank files
        self.blank_files = []

        # Button to add a new species selection
        self.add_species_button = QPushButton("Add Species", self)
        self.add_species_button.setStyleSheet(button_style())

        self.add_species_button.clicked.connect(self.add_species)
        self.layout.addWidget(self.add_species_button)

        # Button for output dir
        if os.getcwd() != "/app":
            self.output_dir = get_app_dir()
            outdir_layout = QHBoxLayout()
            self.output_dir_button = QPushButton("Set output directory", self)
            self.output_dir_button.setStyleSheet(button_style())
            self.output_dir_button.clicked.connect(self.select_directory)
            outdir_layout.addWidget(self.output_dir_button)
            self.layout.addLayout(outdir_layout)


    def add_species(self):
        """Add a new species selection panel."""
        species_layout = QHBoxLayout()

        species_button = QPushButton("Select Species Files (.fcs)", self)
        species_button.setStyleSheet(button_style(font_size=11, padding=3))
        species_button.clicked.connect(lambda: self.select_species_files(species_button))

        species_name_display = QLineEdit(self)
        species_name_display.setPlaceholderText("Species Name")

        species_layout.addWidget(species_button)
        species_layout.addWidget(species_name_display)
        self.layout.addLayout(species_layout)

        # Store the species widgets to access later
        self.species_widgets.append((species_button, species_name_display))


    def select_species_files(self, species_button):
        """Select species files and update the species name field."""
        file_types = "Flow Cytometry Files (*.fcs);;All Files (*)"
        file_types_message = ["Select Species Files", "", file_types]
        original_files, _ = QFileDialog.getOpenFileNames(self, *file_types_message)
        if original_files:
            # Get the corresponding species name input field
            for button, name_display in self.species_widgets:
                if button == species_button:
                    # TODO: Should make sure that the species name is unique
                    species_name, ok = QInputDialog.getText(self, "Species Name", "Enter the name of the species:")
                    if ok:
                        name_display.setText(species_name)

                    # Copy files to working directory and store them
                    species_files = []
                    for file in original_files:
                        dest_file = os.path.join(self.working_directory, os.path.basename(file))
                        shutil.copy(file, dest_file)
                        species_files.append(dest_file)

                    species_button.setText(",".join([os.path.basename(x) for x in species_files]))

                    # Store the species files in the dictionary
                    self.species_files[species_name] = species_files

                    # Optionally parse files to extract metadata, etc.
                    for file in species_files:
                        meta, data = fcsparser.parse(file, reformat_meta=True)
                        if 'Time' in data.columns:
                            data = data.drop(columns=['Time'])
                        print(f"Metadata for {os.path.basename(file)}:")
                        print(meta)
                        print(f"First few rows of data for {os.path.basename(file)}:")
                        print(data.head())
                    break
        else:
            species_button.setText(" ".join([file_types_message[0], "(.fcs)"]))
            for button, name_display in self.species_widgets:
                if button == species_button:
                    name_display.setText("Species Name")


    def select_blank_files(self):
        """Select blank files and copy them to the working directory."""
        file_types = "Flow Cytometry Files (*.fcs);;All Files (*)"
        select_blanks_message = ["Select Blank Files", "", file_types]
        original_files, _ = QFileDialog.getOpenFileNames(self, *select_blanks_message)
        if original_files:
            # Copy files to working directory and parse
            for file in original_files:
                dest_file = os.path.join(self.working_directory, os.path.basename(file))
                shutil.copy(file, dest_file)
                self.blank_files.append(dest_file)

                # Parse the file, drop 'Time' column, and display metadata
                meta, data = fcsparser.parse(dest_file, reformat_meta=True)
                if 'Time' in data.columns:
                    data = data.drop(columns=['Time'])
                print(f"Metadata for {os.path.basename(file)}:")
                print(meta)
                print(f"First few rows of data for {os.path.basename(file)}:")
                print(data.head())

            QMessageBox.information(self, "Files Selected", "Blank files selected successfully.")
            self.blank_button.setText(",".join([os.path.basename(x) for x in self.blank_files]))
        else:
            print(self.blank_files)
            # if len(self.blank_files) == 0:
            print("hello blanks")
            self.blank_button.setText(" ".join([select_blanks_message[0], "(.fcs)"]))


    def select_directory(self):
        descr = "Output directory to save CellScanner findings and intermediate files."
        output_dir = QFileDialog.getExistingDirectory(self, "Set output directory", descr)
        if output_dir:
            self.output_dir = output_dir
            if not os.path.exists(output_dir):
                os.mkdirs(output_dir)
            if self.output_dir != self.working_directory:
                if os.listdir(self.working_directory):
                    shutil.move(self.working_directory, output_dir)
                self.working_directory = os.path.join(output_dir, os.path.basename(self.working_directory))
                os.makedirs(self.working_directory, exist_ok=True)
                if len(self.blank_files) > 0:
                    for index, filename in enumerate(self.blank_files):
                        self.blank_files[index] = os.path.join(self.working_directory, os.path.basename(filename))
                if len(self.species_files) > 0:
                    for species, species_files in self.species_files.items():
                        for index, filename in enumerate(species_files):
                            species_files[index] = os.path.join(self.working_directory, os.path.basename(filename))
                        self.species_files[species] = species_files
            self.output_dir_button.setText(output_dir)
        else:
            self.output_dir_button.setText("Set output directory")
