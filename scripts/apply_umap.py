import os
import umap
import fcsparser
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from .nn import prepare_for_training
from .helpers import Stain, get_stains_from_panel, apply_gating
from .illustrations import umap_plot


def process_file(file, species_name, n_events, stain_1, stain_2, model_dir):          #  , scaling_constant=150
    """
    Processes import .fcs files by first gating (if asked) and then sampling their entries
    to only keep a subset of them for the training step.

    :param file (str): Path to the .fcs file
    :param species_name (str): Name of the species corresponding to the .fcs file
    :param n_events (int): Number of entries of the .fcs file to keep for model training
    :param stain_1 (Strain): User parameters for the (live/dead) staining
    :param stain_2 (Stain): User parammeters for the (cells/debris) staining
    :param model_dir (str): Path for model-related output files to be saved

    :return sampled_df: The gated (if asked) and sampled entris of the .fcs file to be used for the model training
    """
    _, df = fcsparser.parse(file, reformat_meta=True)
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])  # Remove Time column

    print("Processing file: ", species_name)

    if stain_1 is None and stain_2 is None:
        print("No gating for the training step.")

    else:
        # Apply gating
        with open(os.path.join(model_dir, "gating_input_data.txt"), "w") as f:
            # Writing species name and original number of entries
            f.write(f"species name: {species_name}\n")
            f.write(f"original number of entries: {df.shape}\n")

            # Apply gating process
            try:
                gated_df, _ = apply_gating(df, stain_1, stain_2)
            except ValueError as e:
                raise ValueError(f"Error processing species {species_name}: {e}") from e

            # Print and write columns to the file
            f.write(f"df.columns: {df.columns.tolist()}\n")
            f.write(f"gated_df.columns: {gated_df.columns.tolist()}\n")

            # Apply gating for stain 1 if channel is not None
            if isinstance(stain_1, Stain) and stain_1.channel:
                df = df[gated_df["dead"] == False]
                f.write(f"number of entries after gating for stain1: {df.shape}\n")

            # Apply gating for stain 2 if channel is not None
            if isinstance(stain_2, Stain) and stain_2.channel:
                df = df[gated_df["cell"] == True]
                f.write(f"number of entries after gating for stain2: {df.shape}\n")

    # Keep a subset of the entries for the training part
    sampled_df = df.sample(n=min(n_events, len(df)))
    sampled_df['Species'] = species_name

    return sampled_df


def process_files(TrainPanel=None, **kwargs):
    """
    Main function for UMAP application.
    """
    gui = False
    if type(TrainPanel).__name__ == "TrainModelPanel":

        # Read parameters from the GUI
        params = {
            "n_events": int(TrainPanel.event_combo.currentText()),
            "umap_n_neighbors": int(TrainPanel.umap_nneighbors_combo.combo.currentText()),
            "umap_min_dist": float(TrainPanel.umap_mindist_combo.combo.currentText()),
            "nonblank_threshold": int(TrainPanel.nn_nonblank_combo.combo.currentText()),
            "blank_threshold": int(TrainPanel.nn_blank_combo.combo.currentText()),
            "species_files_names_dict": TrainPanel.file_panel.species_files,
            "blank_files": TrainPanel.file_panel.blank_files,
            "working_directory": TrainPanel.file_panel.working_directory,
        }

        gating = TrainPanel.gating_checkbox.isChecked()

        if gating:
            stain_1, stain_2 = get_stains_from_panel(TrainPanel)
        else:
            stain_1, stain_2 = None, None
        gui = True
    else:
        # Read parameters from kwargs
        required_keys = [
            "n_events", "umap_n_neighbors", "umap_min_dist",
            "blank_files", "blank_threshold", "nonblank_threshold",
            "species_files_names_dict", "working_directory",
            "stain_1", "stain_2"
        ]
        params = {key: kwargs[key] for key in required_keys}
        stain_1, stain_2 = params["stain_1"], params["stain_2"]

    # Extract parameters into variables for later use
    n_events = params["n_events"]
    umap_n_neighbors = params["umap_n_neighbors"]
    umap_min_dist = params["umap_min_dist"]
    nonblank_threshold = params["nonblank_threshold"]
    blank_threshold = params["blank_threshold"]
    species_files_names_dict = params["species_files_names_dict"]
    blank_files = params["blank_files"]
    working_directory = params["working_directory"]
    model_dir = os.path.join(working_directory, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Process files for all species dynamically
    all_species_dataframes = []
    for species_name, species_files in species_files_names_dict.items():
        species_dataframes = []
        for sp_file in species_files:
            try:
                species_dataframes.append(
                    process_file(
                        file=sp_file, species_name=species_name,
                        n_events=n_events,
                        stain_1=stain_1, stain_2=stain_2,
                        model_dir=model_dir
                    )
                )
            except Exception as e:
                raise Exception(f"Error while processing species file {species_name}: {e}") from e  # Corrected
        all_species_dataframes.append(species_dataframes)

    # Process blanks
    blank_dataframes = []
    blank_stain = Stain(channel=None, sign=None, value=None)
    for blank_file in blank_files:
        try:
            blank_dataframes.append(
                process_file(
                    file=blank_file, species_name='Blank', n_events=n_events,
                    stain_1=blank_stain, stain_2=blank_stain,
                    model_dir=model_dir
                )
            )
        except Exception as e:
            raise(f"Error processing blank file {blank_file}: {e}") from e

    # Combine data
    combined_df = pd.concat([df for species_dataframes in all_species_dataframes for df in species_dataframes] + blank_dataframes)


    columns_to_plot = combined_df.columns.difference(['Species']).tolist()
    data_subset = combined_df[columns_to_plot].values
    scaled_data_subset = StandardScaler().fit_transform(data_subset)

    # Dimensionality reduction using UMAP
    print("Build UMAP reducer")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist
    )

    # Fit and transform the data
    embedding = reducer.fit_transform(scaled_data_subset)

    label_map = {species_name: idx for idx, species_name in enumerate(species_files_names_dict.keys())}
    label_map['Blank'] = len(label_map)
    mapped_labels = combined_df['Species'].map(label_map).values

    # Plot UMAP before filtering
    umap_plot(combined_df, embedding, model_dir, "Before", None)

    # -----------------------
    # Call nn basic class
    # -----------------------

    # Nearest Neighbors filtering
    print("Instantiate the Nearest Neighbors Model")
    nn = NearestNeighbors(n_neighbors=50)

    print("Fit the Model to the Data")
    nn.fit(embedding)

    print("Find Nearest Neighbors")
    _, indices = nn.kneighbors(embedding)
    indices_to_keep = []

    for i in range(len(embedding)):
        neighbor_labels = mapped_labels[indices[i][1:]] if len(indices[i]) > 1 else []
        if mapped_labels[i] != label_map['Blank']:
            non_blank_neighbors = np.sum(neighbor_labels == mapped_labels[i])
            if non_blank_neighbors >= nonblank_threshold:
                indices_to_keep.append(i)
        else:
            blank_neighbors = np.sum(neighbor_labels == label_map['Blank'])
            if blank_neighbors >= blank_threshold:
                indices_to_keep.append(i)

    cleaned_data = combined_df.iloc[indices_to_keep]

    # Plot UMAP after filtering
    umap_plot(cleaned_data, embedding, model_dir, "After", indices_to_keep)
    print("Data processing and UMAP filtering successful.")
    if gui:
        TrainPanel.cleaned_data = cleaned_data
        prepare_for_training(TrainPanel)

    return cleaned_data
