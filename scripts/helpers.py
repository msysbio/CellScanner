import os
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .illustrations import gating_plot

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


def get_stains_from_panel(Panel):
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
    stain_1 = Panel.stain1_selector.combo.currentText()  # It should be the column name
    if stain_1 != "Not applicable":
        stain1_channel = stain_1
        stain1_relation = Panel.stain1_selector.relation.currentText()
        stain1_threshold = float(Panel.stain1_selector.threshold.text())
        stain1 = Stain(stain1_channel, stain1_relation, stain1_threshold)
    else:
        stain1 = None

    # Stain 2
    stain_2 = Panel.stain2_selector.combo.currentText()  # It should be the column name
    if stain_2 != "Not applicable":
        stain2_channel = stain_2
        stain2_relation = Panel.stain2_selector.relation.currentText()
        stain2_threshold = float(Panel.stain2_selector.threshold.text()) if Panel.stain2_selector.threshold.text() else None
        stain2 = Stain(stain2_channel, stain2_relation, stain2_threshold)
    else:
        stain2 = None

    return stain1, stain2


def stain_sannity_check(df, label, channel, sign, threshold):
    """
    Checks if gating applied for a stain returns both True and False cases.
    If not, raises an error so the user refines their thresholds.
    """
    counts = df[label].value_counts()
    if True not in counts.index or False not in counts.index:
        stain_min, stain_max = np.min(df[channel]), np.max(df[channel])
        raise ValueError(
            f"Invalid gating. Please check the gating thresholds."
            f"Stain {channel} ranges between {stain_min} and {stain_max}, while current gating thresholds are {sign} {threshold}."
        )


def apply_gating(data_df,
                 stain1=None,
                 stain2=None,
                 extra_stains=None
    ):

    all_labels = []

    # Copy the DataFrame to not change the original data
    gated_data_df = data_df.copy()

    # Temporarily remove the 'predictions' column to avoid issues with numeric operations -- irrelevant in TRAINING
    predictions_column = gated_data_df.pop('predictions') if 'predictions' in gated_data_df.columns else None

    # Reintegrate the 'predictions' column after the arcsinh transformation
    if predictions_column is not None:
        gated_data_df['predictions'] = predictions_column

    if stain1 is not None:

        if stain1.channel is not None and stain1.channel != "Not applicable":

            # Initialize the 'state' column with 'not dead'
            gated_data_df['dead'] = False
            # Apply gating based on the first stain (live/dead)
            if stain1.sign in ['>', 'greater_than']:
                gated_data_df.loc[gated_data_df[stain1.channel] > stain1.value, 'dead'] = True
            elif stain1.sign in ['<', 'less_than']:
                gated_data_df.loc[gated_data_df[stain1.channel] < stain1.value, 'dead'] = True
            # Sannity check
            try:
                stain_sannity_check(gated_data_df, "dead", stain1.channel, stain1.sign, stain1.value)
                all_labels.append("dead")
            except ValueError as e:
                raise ValueError(f"Gating failed for stain1: {e}") from e  # Preserve original traceback

    if stain2 is not None:
        if stain2.channel is not None and stain2.channel != "Not applicable":

            # Initialize the 'state' column with 'not dead'
            gated_data_df['cell'] = False
            # Apply gating based on the first stain (live/dead)
            if stain2.sign in ['>', 'greater_than']:
                gated_data_df.loc[gated_data_df[stain2.channel] > stain2.value, 'cell'] = True
            elif stain2.sign in ['<', 'less_than']:
                gated_data_df.loc[gated_data_df[stain2.channel] < stain2.value, 'cell'] = True
            # Sannity check
            try:
                stain_sannity_check(gated_data_df, "cell", stain2.channel, stain2.sign, stain2.value)
                all_labels.append("cell")
            except ValueError as e:
                raise ValueError(f"Gating failed for stain2: {e}") from e  # Preserve original traceback

    # Apply gating based on the second stain (debris)
    if stain1 is not None and stain2 is not None:
        if stain2.channel and stain2.value and stain1.channel:

            gated_data_df["state"] = "debris"

            gated_data_df["state"].loc[
                (gated_data_df["dead"] == False) & (gated_data_df["cell"] == True)
            ] = "live"

            gated_data_df["state"].loc[
                (gated_data_df["dead"] == True) & (gated_data_df["cell"] == True)
            ] = "inactive"

            all_labels.append("state")

    # Apply gating on extra stains
    if extra_stains is not None:
        for channel, details in extra_stains.items():
            sign, threshold, label = details
            # Create the comparison operator dynamically
            condition = gated_data_df[channel] > threshold if sign == ">" else gated_data_df[channel] < threshold
            gated_data_df[label] = condition
            try:
                stain_sannity_check(gated_data_df, label, channel, sign, threshold)
            except ValueError as e:
                raise ValueError(f"Gating failed for extra stain: {e}") from e  # Preserve original traceback

            all_labels.append(label)

    return gated_data_df, all_labels



def save_gating_results(gated_data_df, output_dir, sample, x_axis, y_axis, z_axis, all_labels):
    """

    """
    # Create a directory for gating results
    gated_dir = os.path.join(output_dir, 'gated')
    os.makedirs(gated_dir, exist_ok=True)

    # Initialize an empty DataFrame to hold all state counts
    combined_counts_df = pd.DataFrame()

    # Iterate over each species and calculate the state counts
    species_names = gated_data_df['predictions'].unique()
    if "state" in all_labels:
        all_labels.remove("dead") ; all_labels.remove("cell")

    for species in species_names:
        species_df = pd.DataFrame()
        for label in all_labels:
            if label == "state":
                s = gated_data_df[gated_data_df['predictions'] == species][label].value_counts()
            else:
                s = gated_data_df[gated_data_df['predictions'] == species][label].value_counts()
                if s.index[0] == True:
                    s.index = [label, "_".join(["not", label])] if len(s.index) == 2 else [label]
                else:
                    s.index = ["_".join(["not", label]), label] if len(s.index) == 2 else ["_".join(["not", label])] # Default for False case
            s.name = species
            species_df = pd.concat([species_df, s], axis=0)

        combined_counts_df = pd.concat([combined_counts_df, species_df], axis=1)

    # Save the combined state counts to a single CSV file
    combined_counts_df.to_csv(
        os.path.join(gated_dir, "_".join([sample,'gating.csv']))
    )

    # Plot status if both stains provided
    gating_plot(gated_data_df, species_names, x_axis, y_axis, z_axis, gated_dir, sample, all_labels)

    print("3D scatter plot for gated data saved to:", gated_dir)


@dataclass
class Stain:
    channel: str
    sign: str
    value: float
    label: Optional[str] = None
