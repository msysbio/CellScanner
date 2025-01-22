import os
import umap
import fcsparser
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from .nn import prepare_for_training
from .helpers import get_abs_path


def process_file(file, species_name, n_events):

    _, df = fcsparser.parse(file, reformat_meta=True)
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])  # Remove Time column
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
            "umap_n_neighbors": int(TrainPanel.umap_nneighbors_combo.currentText()),
            "umap_min_dist": float(TrainPanel.umap_mindist_combo.currentText()),
            "nonblank_threshold": int(TrainPanel.nn_nonblank_combo.currentText()),
            "blank_threshold": int(TrainPanel.nn_blank_combo.currentText()),
            "species_files_names_dict": TrainPanel.file_panel.species_files,
            "blank_files": TrainPanel.file_panel.blank_files,
            "working_directory": TrainPanel.file_panel.working_directory
        }
        gui = True
    else:
        # Read parameters from kwargs
        required_keys = [
            "n_events", "umap_n_neighbors", "umap_min_dist",
            "blank_files", "blank_threshold", "nonblank_threshold",
            "species_files_names_dict", "working_directory"
        ]
        params = {key: kwargs[key] for key in required_keys}

    # Extract parameters into variables for later use
    n_events = params["n_events"]
    umap_n_neighbors = params["umap_n_neighbors"]
    umap_min_dist = params["umap_min_dist"]
    nonblank_threshold = params["nonblank_threshold"]
    blank_threshold = params["blank_threshold"]
    species_files_names_dict = params["species_files_names_dict"]
    blank_files = params["blank_files"]
    working_directory = params["working_directory"]

    # Process files for all species dynamically
    all_species_dataframes = []
    for species_name, species_files in species_files_names_dict.items():
        species_dataframes = []
        for sp_file in species_files:
            try:
                species_dataframes.append(
                    process_file(file=sp_file, species_name=species_name, n_events=n_events)
                )
            except Exception as e:
                print(f"Error processing file {sp_file}: {e}")
        all_species_dataframes.append(species_dataframes)

    # Process blanks
    blank_dataframes = []
    for blank_file in blank_files:
        try:
            blank_dataframes.append(
                process_file(file=blank_file, species_name='Blank', n_events=n_events)
            )
        except Exception as e:
            print(f"Error processing blank file {blank_file}: {e}")

    # Combine data
    combined_df = pd.concat([df for species_dataframes in all_species_dataframes for df in species_dataframes] + blank_dataframes)

    columns_to_plot = combined_df.columns.difference(['Species']).tolist()
    data_subset = combined_df[columns_to_plot].values
    scaled_data_subset = StandardScaler().fit_transform(data_subset)

    # Dimensionality reduction using UMAP
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist
    )

    embedding = reducer.fit_transform(scaled_data_subset)

    label_map = {species_name: idx for idx, species_name in enumerate(species_files_names_dict.keys())}
    label_map['Blank'] = len(label_map)
    mapped_labels = combined_df['Species'].map(label_map).values

    # Generate UMAP plot before filtering
    fig_before = px.scatter_3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        color=combined_df['Species'],
        title='UMAP Projection Before Filtering',
        opacity=0.7,
        labels={
            'x': 'UMAP1',
            'y': 'UMAP2',
            'z': 'UMAP3',
            'color': 'Species'
        },

    )
    fig_before.update_traces(marker=dict(size=5))

    # Define the path to save the plot
    model_dir = os.path.join(working_directory, "model")  # get_abs_path('model/statistics')
    os.makedirs(model_dir, exist_ok=True)
    umap_before_path = os.path.join(model_dir, 'umap_before_filtering.html')

    # Save the plot as an HTML file
    fig_before.write_html(umap_before_path)

    # Nearest Neighbors filtering
    nn = NearestNeighbors(n_neighbors=50)
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)
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

    # Generate UMAP plot after filtering
    fig_after = px.scatter_3d(
        x=embedding[indices_to_keep, 0],
        y=embedding[indices_to_keep, 1],
        z=embedding[indices_to_keep, 2],
        color=cleaned_data['Species'],
        title='UMAP Projection After Filtering',
        opacity=0.7,
        labels={
            'x': 'UMAP1',
            'y': 'UMAP2',
            'z': 'UMAP3',
            'color': 'Species'
        },

    )
    fig_after.update_traces(marker=dict(size=5))

    umap_after_path = os.path.join(model_dir, 'umap_after_filtering.html')
    fig_after.write_html(umap_after_path)

    print("Data processing and UMAP filtering successful.")

    if gui:
        print("Run prepare for training function.")
        TrainPanel.cleaned_data = cleaned_data
        # return prepare_for_training(TrainPanel)
        prepare_for_training(TrainPanel)

    return cleaned_data

