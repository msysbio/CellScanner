"""
The :class:`ImportFilePanel` class provides a user interface panel for selecting and managing flow cytometry files
for monoculture species 1, monoculture species 2, and blank samples. The selected files are copied to a
working directory, where they can be processed by other components of the application.

Key Features:

- Allows users to select ``.fcs`` files for species 1, species 2, and blank samples.
- Files are copied to a working directory to ensure the original files remain unchanged.
- Displays metadata and the first few rows of data for each selected file.
- Handles file parsing using the `fcsparser` library.
- Supports cleanup of the working directory to remove copied files.

"""
import os
import shutil
import fcsparser

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, \
    QFileDialog, QInputDialog, QMessageBox


from .helpers import get_app_dir, time_based_dir, load_model_from_files
from .GUIhelpers import button_style, LabeledSpinBox, _GuiMessages, load_fcs_file

"""
Authors:
    - Ermis Ioannis Michail Delopoulos
    - Haris Zafeiropoulos

Date: 2024-2025
"""

class ImportFilePanel(QWidget):
    """
    A QWidget subclass that provides the interface for selecting files and managing the working directory.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.model = None

        # Working directory for copies
        self.working_directory = time_based_dir(base_path=get_app_dir(), prefix='working_files')
        os.makedirs(self.working_directory, exist_ok=True)  # Ensure the directory exists

        # List to store species-related file paths and names
        self.species_widgets = []
        self.species_files = {}  # A dictionary to store species and their corresponding files

        self.file_types = "Flow Cytometry Files (*.fcs);;All Files (*)"
        self.select_blanks_message = ["Select Blank Files", "", self.file_types]

        # Blank Files Selection
        blank_layout = QHBoxLayout()
        self.blank_button = QPushButton("Select Blank Files (.fcs)", self)
        self.blank_button.setStyleSheet(button_style())
        self.blank_button.setToolTip(_GuiMessages.BLANKS_MULTIFILES)
        self.blank_button.clicked.connect(self.select_blank_files)
        blank_layout.addWidget(self.blank_button)
        self.layout.addLayout(blank_layout)

        # storing blank files
        self.blank_files = []

        # Button to add a new species selection
        self._init_species_layout()

        # Previously trained model button
        self.previously_trained_model_button = QPushButton("Add Model", self)
        self.previously_trained_model_button.setStyleSheet(button_style(bck_col="#f7c67d", bck_col_hov="#deb270"))
        self.previously_trained_model_button.setToolTip(_GuiMessages.PREVIOUSLY_TRAINED_MODEL)
        self.previously_trained_model_button.clicked.connect(self.add_prev_trained_model)
        self.layout.addWidget(self.previously_trained_model_button)

        # Scaling constant of previously trained model
        self.scaling_constant = LabeledSpinBox(
            "Scaling Constant:",
            min_value=0, max_value=1000, step=1, default_value=150
        )
        self.layout.addWidget(self.scaling_constant)

        # Model load
        self.model_loaded = False
        self._toggle_const_scaling()

        # Button for output dir
        if os.getcwd() != "/app":
            self.output_dir = get_app_dir()
            outdir_layout = QHBoxLayout()
            self.output_dir_button = QPushButton("Set output directory", self)
            self.output_dir_button.setStyleSheet(button_style(bck_col="#f7c67d", bck_col_hov="#deb270"))
            self.output_dir_button.setToolTip(_GuiMessages.OUTPUT_DIR)
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

        file_types_message = ["Select Species Files", "", self.file_types]
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
                    for orig_file in original_files:
                        dest_file = os.path.join(self.working_directory, os.path.basename(orig_file))
                        shutil.copy(orig_file, dest_file)
                        species_files.append(dest_file)

                    species_button.setText(",".join([os.path.basename(x) for x in species_files]))

                    # Store the species files in the dictionary
                    self.species_files[species_name] = species_files

                    # Optionally parse files to extract metadata, etc.
                    for sp_file in species_files:
                        _, data = fcsparser.parse(sp_file, reformat_meta=True)
                        if 'Time' in data.columns:
                            data = data.drop(columns=['Time'])

                    break  # IPORTANT: Breaks the loop over the buttons
        else:
            species_button.setText(" ".join([file_types_message[0], "(.fcs)"]))
            for button, name_display in self.species_widgets:
                if button == species_button:
                    name_display.setText("Species Name")

        print("Species (monoculture) files:", self.species_files)


    def select_blank_files(self):
        """Select blank files and copy them to the working directory."""
        original_files, _ = QFileDialog.getOpenFileNames(self, *self.select_blanks_message)
        if original_files:
            print("Blanks:", original_files)
            blank_files = []
            # Copy files to working directory and parse
            for file in original_files:
                dest_file = os.path.join(self.working_directory, os.path.basename(file))
                shutil.copy(file, dest_file)
                blank_files.append(dest_file)

            # Keep the numeric columns to be used in the TrainModelPanel in case user applies line gating
            _, _, numeric_columns = load_fcs_file(original_files)
            self.numeric_colums_set = set(numeric_columns)

            # Blank filenames
            self.blank_files = blank_files
            QMessageBox.information(self, "Files Selected", "Blank files selected successfully.")
            self.blank_button.setText(",".join([os.path.basename(x) for x in self.blank_files]))

        else:
            print(self.blank_files)
            self.blank_button.setText(" ".join([self.select_blanks_message[0], "(.fcs)"]))


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


    def add_prev_trained_model(self):
        descr = "Directory with previously trained model. The folder needs to include all the following three files:  "
        trained_model_dir = QFileDialog.getExistingDirectory(self, "Set directory of previously trained model", descr)
        if trained_model_dir:
            try:
                self.model, self.scaler, self.le = load_model_from_files(trained_model_dir)
                QMessageBox.information(self, "Model loading", "Model files loaded successfully.")
                self.previously_trained_model_button.setText(trained_model_dir)
                self.model_loaded = True if self.model is not None else False
                self.blank_files = []
                self.species_files = {}
            except:
                QMessageBox.information(self, "Model loading", "Model files were not loaded!")
            self._toggle_const_scaling()


    def _toggle_const_scaling(self):
        # Show or hide scaling constant layout based on model_loaded flag
        self.model_loaded = True if self.model is not None else False
        self.scaling_constant.label.setVisible(self.model_loaded)
        self.scaling_constant.setVisible(self.model_loaded)


    def _init_species_layout(self):
        self.species_layout = QHBoxLayout()
        self.add_species_button = QPushButton("Add Species", self)
        self.add_species_button.setStyleSheet(button_style())
        self.add_species_button.setToolTip("You may add one or several .fcs files for the same species.")
        self.add_species_button.clicked.connect(self.add_species)
        self.species_layout.addWidget(self.add_species_button)
        self.layout.addLayout(self.species_layout)
