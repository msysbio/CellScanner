import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Optional


def get_app_dir():
    """Get absolute path relative to the executable location."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path


def get_abs_path(relative_path):
    """Get absolute path to a resource, relative to the base directory."""
    return os.path.join(get_app_dir(), relative_path)


def time_based_dir(prefix, base_path, multiple_cocultures=False):
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    time_dir_name = "_".join([prefix, timestamp])
    if os.getcwd() == "/app":
        time_dir = os.path.join("/media", time_dir_name)
    else:
        time_dir = os.path.join(base_path, time_dir_name)
    if os.path.exists(time_dir) and multiple_cocultures is False:
        base, ext = os.path.splitext(time_dir)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        time_dir = f"{base}_{counter}{ext}"
    return time_dir


def button_style(font_size=12, padding=5, color="black", bck_col="#90EE90",
                 bck_col_hov="#7FCF7F", bck_col_clicked="#72B572", radius=5):
    style = f"""
    QPushButton {{
        font-size: {font_size}px;
        font-weight: bold;
        padding: {padding}px;
        color: {color};
        background-color: {bck_col};  /* Light green color */
        border-radius: {radius}px;
    }}
    QPushButton:hover {{
        background-color: {bck_col_hov};  /* Slightly darker green on hover */
    }}
    QPushButton:pressed {{
        background-color: {bck_col_clicked};  /* Even darker when pressed */
    }}
    """
    return style


def load_model_from_files(trained_model_dir):

    print("Loading model from files")
    from tensorflow.keras.models import load_model
    import joblib

    modelfiles = ["trained_model.keras", "scaler.pkl", "label_encoder.pkl"]
    model_path, scaler_path, le_path = [os.path.join(trained_model_dir, x) for x in modelfiles]

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(le_path)
        return model, scaler, label_encoder

    except Exception as e:
        print(f"Error loading model or preprocessing objects: {e}")
        raise ValueError(f"No valid model directory. Check whether all 3 required files are there and valid.")


def create_file_path(output_dir, sample, name, extension):
    """Helper function to create file paths."""
    if sample:
        return os.path.join(output_dir, f"{sample}_{name}.{extension}")
    return os.path.join(output_dir, f"{name}.{extension}")


def get_stains_from_panel(PredictionPanel):
    """
    Build Stain instances for the two main stain types of living/dead and cells/not cells cases.
    In this case, no label is part of the Stain instance.
    Function to be used only in the GUI framework.

    Arguments:
        PredictionPanel:
    Returns:
        stain1 (Stain)
        stain2 (Stain)
    """
    # Stain 1
    stain_1 = PredictionPanel.stain1_combo.currentText()  # It should be the column name
    if stain_1 != "Not applicable":
        stain1_channel = stain_1
        stain1_relation = PredictionPanel.stain1_relation.currentText()
        stain1_threshold = float(PredictionPanel.stain1_threshold.text())
        stain_1 = Stain(stain1_channel, stain1_relation, stain1_threshold)
    else:
        stain1 = None
    # Stain 2
    stain_2 = PredictionPanel.stain2_combo.currentText()  # It should be the column name
    if stain_2 != "Not applicable":
        stain2_channel = stain_2
        stain2_relation = PredictionPanel.stain2_relation.currentText()
        stain2_threshold = float(PredictionPanel.stain2_threshold.text()) if PredictionPanel.stain2_threshold.text() else None
        stain2 = Stain(stain2_channel, stain2_relation, stain2_threshold)
    else:
        stain2 = None
    return stain1, stain2


@dataclass
class Stain:
    channel: str
    sign: str
    value: float
    label: Optional[str] = None
