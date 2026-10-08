"""
Helpers functions to support CellScanner main tasks.
"""

import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .illustrations import gating_plot

NOT_APPLICABLE = "Not applicable"


@dataclass
class Stain:
    channel: str
    sign: str
    value: float
    label: str | None = None


def get_app_dir():
    """Get absolute path relative to the executable location."""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path


def get_abs_path(relative_path):
    """Get absolute path to a resource, relative to the base directory."""
    return os.path.join(get_app_dir(), relative_path)


def time_based_dir(prefix, base_path, multiple_cocultures=False):
    timestamp = datetime.now().astimezone().strftime("%d-%m-%Y_%H-%M")
    time_dir_name = f"{prefix}_{timestamp}"
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


def load_model_from_files(trained_model_dir):
    """
    Loads a previously trained model from CellScanner.

    :param trained_model_dir: Path to the directory with the previously trained model files. Under trained_model_dir,
                                the user needs to make sure there are all three following files: "trained_model.keras", "scaler.pkl", "label_encoder.pkl"
    :raises ValueError: If any of the 3 required files is missing.
    """

    import joblib
    from tensorflow.keras.models import load_model

    modelfiles = ["trained_model.keras", "scaler.pkl", "label_encoder.pkl"]
    model_path, scaler_path, le_path = [
        os.path.join(trained_model_dir, x) for x in modelfiles
    ]

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(le_path)
        return model, scaler, label_encoder

    except Exception as e:
        raise ValueError(
            f"No valid model directory. Check whether all 3 required files are there and valid.error: {e}"
        ) from e


def create_file_path(output_dir, sample, name, extension):
    """Helper function to create file paths."""
    if sample:
        return os.path.join(output_dir, f"{sample}_{name}.{extension}")
    return os.path.join(output_dir, f"{name}.{extension}")


def get_channels(channels_df):
    """
    : channels_df: A pd.DataFrame part of the fcsparser loading function

    """
    channels_df["long_channel"] = channels_df.apply(
        lambda row: (
            f"{row['$PnN']} [{row['$PnS']}]"
            if row["$PnN"] != row["$PnS"]
            else row["$PnN"]
        ),
        axis=1,
    )
    channels = set(channels_df["long_channel"])
    return channels


def gate(series: pd.Series, sign: str, value: float) -> pd.Series:
    """
    Returns a boolean mask of the entries of ``series`` that meet the threshold.
    Accepts both the GUI (``>``, ``<``) and the CLI (``greater_than``, ``less_than``) sign spellings.
    """
    if sign in (">", "greater_than"):
        return series > value
    if sign in ("<", "less_than"):
        return series < value
    raise ValueError(
        f"Unknown sign {sign!r}; use '>'/'greater_than' or '<'/'less_than'."
    )


def get_stains_from_panel(Panel):
    """
    Build Stain instances for the two main stain types of living/dead and cells/not cells cases.
    In this case, no label is part of the Stain instance.
    Function to be used only in the GUI framework.

    Arguments:
        Panel (:class:`PredictionPanel` | :class:`TrainModelPanel`):
    Returns:
        stain1 (Stain)
        stain2 (Stain)
    """
    # Stain 1

    stain_1 = Panel.stain1_selector.combo.currentText()  # It should be the column name

    if stain_1 != NOT_APPLICABLE:
        match = re.search(
            r"\[(.*?)\]", stain_1
        )  # In case we chose a channel with a second name in brackets
        stain1_channel = match.group(1) if match else stain_1
        stain1_relation = Panel.stain1_selector.relation.currentText()
        stain1_threshold = float(Panel.stain1_selector.threshold.text())

        stain1 = Stain(stain1_channel, stain1_relation, stain1_threshold)

    else:
        stain1 = Stain(channel=None, sign=None, value=None)

    # Stain 2
    stain_2 = Panel.stain2_selector.combo.currentText()  # It should be the column name

    if stain_2 != NOT_APPLICABLE:
        match = re.search(r"\[(.*?)\]", stain_2)
        stain2_channel = match.group(1) if match else stain_2
        stain2_relation = Panel.stain2_selector.relation.currentText()
        stain2_threshold = (
            float(Panel.stain2_selector.threshold.text())
            if Panel.stain2_selector.threshold.text()
            else None
        )
        stain2 = Stain(stain2_channel, stain2_relation, stain2_threshold)

    else:
        stain2 = Stain(channel=None, sign=None, value=None)

    return stain1, stain2


def extact_channel(long_channel):
    match = re.search(
        r"\[(.*?)\]", long_channel
    )  # In case we chose a channel with a second name in brackets
    channel = match.group(1) if match else long_channel
    return channel


def stain_sannity_check(df, label, channel, sign, threshold):
    """
    Checks if gating applied for a stain returns both True and False cases.
    If not, warns the user so they can refine their thresholds; a gate that does not split
    a file is not necessarily an error (e.g. a clean monoculture above a debris threshold).
    """
    frac = df[label].mean()
    if frac in (0.0, 1.0):
        stain_min, stain_max = np.min(df[channel]), np.max(df[channel])
        warnings.warn(
            f"Gate {channel} {sign} {threshold} labels {frac:.0%} of the events as '{label}'. "
            f"Stain {channel} ranges between {stain_min} and {stain_max}."
        )


def apply_gating(
    data_df: pd.DataFrame,
    stain1: Stain = None,
    stain2: Stain = None,
    extra_stains: dict = None,
):
    """
    Applies line gating to a dataset based on fluorescence or marker intensity thresholds.

    This function evaluates whether the values of specified stains (channels) in the dataset
    meet their respective thresholds. It assigns labels (True or False) based on the gating
    criteria.

    Args:
        data_df (:class:`pandas.DataFrame`): The dataframe containing intensity values for different stains.
        stain1 (Stain, optional): An instance of the :class:`Stain` class representing the first stain used for gating.
        stain2 (Stain, optional): An instance of the :class:`Stain` class representing the second stain used for gating.
        extra_stains (Dict, optional): A dictionary with the label of a stain as key, and its channel, sign, and threshold as their value for multi-channel gating.

    Returns:
        pandas.DataFrame: A dataframe with the gating results, where each row is labeled as True or False.
        list: A list of labels assigned to each row based on the applied gating criteria.

    Example:
        ```
        gated_df, labels = apply_gating(data, stain1=Stain("CD3", threshold=500), stain2=Stain("CD19", threshold=200))
        ```
    """

    all_labels = []

    # Copy the DataFrame to not change the original data
    gated_data_df = data_df.copy()

    # Temporarily remove the 'predictions' column to avoid issues with numeric operations -- irrelevant in TRAINING
    predictions_column = (
        gated_data_df.pop("predictions")
        if "predictions" in gated_data_df.columns
        else None
    )

    # Reintegrate the 'predictions' column after the arcsinh transformation
    if predictions_column is not None:
        gated_data_df["predictions"] = predictions_column

    if stain1.channel is not None:
        """ STAIN FOR CELLS / DEBRIS (sybr green) """
        if stain1.channel is not None and stain1.channel != NOT_APPLICABLE:
            # Entries where the threshold holds are cells; the rest are debris
            gated_data_df["cell"] = gate(
                gated_data_df[stain1.channel], stain1.sign, stain1.value
            )
            stain_sannity_check(
                gated_data_df, "cell", stain1.channel, stain1.sign, stain1.value
            )
            all_labels.append("cell")

    if stain2.channel is not None:
        """ STAIN FOR LIVE / DEAD (PI) """
        if stain2.channel is not None and stain2.channel != NOT_APPLICABLE:
            # Entries where the threshold holds are dead
            gated_data_df["dead"] = gate(
                gated_data_df[stain2.channel], stain2.sign, stain2.value
            )
            stain_sannity_check(
                gated_data_df, "dead", stain2.channel, stain2.sign, stain2.value
            )
            all_labels.append("dead")

    # Apply gating on extra stains
    if extra_stains is not None:
        for channel, details in extra_stains.items():
            sign, threshold, label = details
            gated_data_df[label] = gate(gated_data_df[channel], sign, threshold)
            stain_sannity_check(gated_data_df, label, channel, sign, threshold)
            all_labels.append(label)

    return gated_data_df, all_labels


def save_gating_results(
    gated_data_df, output_dir, sample, x_axis, y_axis, z_axis, all_labels
):
    """
    Counts entries in a dataframe for its of the labels in all_labels and exports those in a csv file.
    It then calls for the :func:`gating_plot` function to export relative visual components.

    :param gated_data_df: A dataframe with a 'predictions' column with the species predicted name label columns (True/False) to count
    :param output_dir: Path where results will be saved
    :param sample: Sample name
    :param x_axis: Name of the X-axis to be plotted (channel among those on the .fcs file)
    :param y_axis: Name of the X-axis to be plotted (channel among those on the .fcs file)
    :param z_axis: Name of the X-axis to be plotted (channel among those on the .fcs file)
    :param all_labels: A list with all the labels for the the stains provided
    """

    # Create a directory for gating results
    gated_dir = os.path.join(output_dir, "gated")
    os.makedirs(gated_dir, exist_ok=True)

    # Iterate over each species and calculate the state counts
    species_names = list(gated_data_df["predictions"].unique())

    gated_data_df.to_csv(os.path.join(gated_dir, f"{sample}_raw_gating.csv"))
    print(
        f"File with gating raw findings for sample {sample} saved at:\n",
        os.path.join(gated_dir, f"{sample}_raw_gating.csv"),
    )
    # Plot status if both stains provided
    gating_plot(
        gated_data_df,
        species_names,
        x_axis,
        y_axis,
        z_axis,
        gated_dir,
        sample,
        all_labels,
    )
    print("3D scatter plot for gated data saved to:", gated_dir)


def merge_prediction_results(output_dir, prediction_type):
    """
    Merge prediction and uncertainty output files into a single file for each case when multiple coculture files are provided.

    :param output_dir: Output directory where CellScanner prediction files were saved
    :param prediction_type: Type of CellScanner output file; 'prediction' (counts) or 'uncertainty' (heretogeneity)
    """

    if prediction_type not in ["prediction", "uncertainty"]:
        raise ValueError(
            "Please provide a valide prediction_type: 'prediction|uncertainty'"
        )

    if prediction_type == "prediction":
        patterns = [
            f"{prediction_type}_counts",
            "state_counts",
            "dead_counts",
            "cell_counts",
        ]

        # Loop through all files in the directory
        dfs = []
        for file_name in os.listdir(output_dir):
            matched_pattern = next(
                (pattern for pattern in patterns if pattern in file_name), None
            )
            if matched_pattern is None:
                continue  # Skip files that don't match any pattern

            # Read each file as a DataFrame
            file_path = os.path.join(output_dir, file_name)
            df = pd.read_csv(
                file_path, index_col=0
            )  #  Make sure you keep first column as index of the dataframe

            # Name the "count" column based on the filename (without extension)
            new_column_name = file_name.split(matched_pattern)[0][:-1]
            df.columns = [new_column_name]
            dfs.append(df)

            pattern = matched_pattern

        # Merge all DataFrames on the "predictions" column
        result = pd.concat(dfs, axis=1)
        result = result.dropna(how="all")
        unknonws = [x for x in result.index if "Unknown" in x]
        if len(unknonws) > 0:
            sum_unknowns = result.loc[unknonws].sum()
            result = result.drop(index=unknonws)
            result.loc["Unknown"] = sum_unknowns
    else:
        pattern = "heterogeneity_results"
        output_dir = os.path.join(output_dir, "heterogeneity_results")

        # Loop through all files in the directory
        dfs = []
        for file_name in os.listdir(output_dir):
            if pattern not in file_name:
                continue
            file_path = os.path.join(output_dir, file_name)

            # Read each file as a DataFrame; run_heterogeneity() writes them tab-separated
            df = pd.read_csv(file_path, sep="\t", index_col="Species")

            # Prefix each metric column with the sample name (filename without the pattern)
            new_column_name = file_name.split(pattern)[0][:-1]
            df.columns = [f"{new_column_name}_{c}" for c in df.columns]
            dfs.append(df)

        # Merge all DataFrames on the species index
        result = pd.concat(dfs, axis=1)

    # Save the final result to a CSV file
    try:
        merged_filename = f"merged_{pattern}.csv"
        merged_file = os.path.join(output_dir, merged_filename)
        result.to_csv(merged_file, index=True)
    except Exception as e:
        print(
            f"No merging case ({e}). Please go through the output files of each sample."
        )
