#!/usr/bin/env python
"""
CellScanner Command Line Interface main class.

Besides using CellScanner throught its GUI, you may use it through a CLI.
To this end, you should first complete a [`config.yml`](../config.yml) file, providing the necessary parameters.
"""
import os
import sys
import yaml
import argparse
import fcsparser
from collections import defaultdict

# Load CellScanner features
from scripts.run_prediction import predict
from scripts.apply_umap import process_files
from scripts.nn import prepare_for_training, train_neural_network
from scripts.helpers import get_app_dir, time_based_dir, load_model_from_files, merge_prediction_results, Stain

class CellScannerCLI():

    def __init__(self, args):
        # Initialize model-related attributes
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.conf = args.config
        self.container = False

        # Load config file
        conf = load_yaml(args.config)

        # Output dir
        outdir = conf.get("output_directory").get("path")
        if os.getcwd() == "/app":
            self.output_dir = "/media"
            self.container = True
        elif outdir is None:
            self.output_dir = get_app_dir()
        else:
            self.output_dir = outdir

        # PREVIOUSLY TRAINED MODEL
        self.prev_trained_model = get_param_value("prev_trained_model", conf)
        if self.prev_trained_model is not None:
            self.model, self.scaler, self.le = load_model_from_files(self.prev_trained_model)
            self.scaling_constant = get_param_value("scaling_constant", conf)

        # TRAINING NEW MODEL
        else:
            # Blank files
            blanks_dir = conf.get("blank_files").get("directories")
            self.blank_files = parse_dicts(blanks_dir, entity="blank_files")

            # Species files
            species_directories = conf.get("species_files").get("directories")
            self.all_species_files, self.all_species = parse_dicts(
                species_directories,
                entity="species_files",
                names="species_names"
            )

            # Training parameters
            self.events = get_param_value("umap_events", conf)
            self.n_neighbors = get_param_value("n_neighbors", conf)
            self.umap_min_dist = get_param_value("umap_min_dist", conf)
            self.nn_blank = get_param_value("nn_blank", conf)
            self.nn_non_blank = get_param_value("nn_non_blank", conf)
            self.folds = get_param_value("folds", conf)
            self.epochs = get_param_value("epochs", conf)
            self.batch_size = get_param_value("batch_size", conf)
            self.early_stopping_patience = get_param_value("early_stopping_patience", conf)
            self.scaling_constant = get_param_value("scaling_constant", conf)

        # Coculture files
        coc_directories = conf.get("coculture_files").get("directories")
        self.coculture_files = parse_dicts(coc_directories, "coculture_files")

        # Prediction parameters
        self.x_axis = get_param_value("x_axis", conf)
        self.y_axis = get_param_value("y_axis", conf)
        self.z_axis = get_param_value("z_axis", conf)
        self.filter_out_uncertain = get_param_value("filter_out_uncertain", conf)
        if self.filter_out_uncertain:
            self.uncertainty_threshold = conf.get("filter_out_uncertain", {}).get("threshold")
        else:
            self.uncertainty_threshold = None

        # Gating parameters
        self.gating = get_param_value("gating", conf)
        self.stain1_train, self.stain2_train = None, None
        self.stain1_predict, self.stain2_predict, self.extra_stains = None, None, None
        if self.gating:

            # Training
            self.stain1_train = get_stain_params("stain1_train", conf)
            self.stain2_train = get_stain_params("stain2_train", conf)

            # Predict
            self.stain1_predict = get_stain_params("stain1_predict", conf)
            self.stain2_predict = get_stain_params("stain2_predict", conf)

            # Extra stains for predict
            self.extra_stains = get_extra_stains(conf)

            self._channel_sannity_check()

    def _channel_sannity_check(self):
        """
        Checks whether a channel provided by the user is actually amont those on the .fcs files.
        """
        basic_stains = [self.stain1_train, self.stain2_train, self.stain1_predict, self.stain2_predict]

        if self.prev_trained_model is None:
            _, data_df = fcsparser.parse(list(self.blank_files)[0], reformat_meta=True)
        else:
            _, data_df = fcsparser.parse(list(self.coculture_files)[0], reformat_meta=True)

        all_channels = data_df.columns

        for stain in basic_stains:
            if stain.channel is not None and stain.channel not in all_channels:
                raise ValueError(f"Channel provided for gating {stain.channel} not present in the .fcs files provided.")

        for stain in self.extra_stains:
            if stain not in all_channels:
                raise ValueError(f"Channel provided for gating {stain.channel} not present in the .fcs files provided.")

        print("Valid channel names.")

    def train_model(self):
        """
        A wrapper for the main training model - related CellScanner functions.
        """
        print("\nAbout to preprocess input files.")
        cleaned_data = process_files(
            n_events = self.events, umap_n_neighbors=self.n_neighbors,
            umap_min_dist=self.umap_min_dist, nonblank_threshold=self.nn_non_blank,
            blank_threshold=self.nn_blank, species_files_names_dict=self.all_species,
            blank_files=self.blank_files, working_directory=self.output_dir,
            stain_1=self.stain1_train, stain_2=self.stain2_train
        )
        print("Files processed. Preparing for training:")
        X_whitened, y_categorical, self.scaler, self.le = prepare_for_training(
            cleaned_data=cleaned_data,
            scaler=self.scaler,
            le=self.label_encoder,
            scaling_constant=self.scaling_constant,
            working_directory=self.output_dir
        )
        print("Read to train the model..")
        self.model, self.cs_uncertainty_threshold = train_neural_network(
            fold_count=self.folds,
            epochs=self.epochs,
            batch_size=self.batch_size,
            patience=self.early_stopping_patience,
            X=X_whitened,
            y=y_categorical,
            species_names=self.le.classes_,
            working_directory=self.output_dir
        )
        print("Model complete!")

    def predict_coculture(self):
        """
        A wrapper for the running prediction step - related CellScanner functions.
        In case where several co-culture files have been provided (samples), CellScanner makes its prediction per sample
        and in the end merges them in a single file.
        """
        print("About to start predicting co-culture profiles.")

        if not all([self.model, self.scaler, self.le]):
            raise ValueError("Please ensure a trained model, scaler, and label encoder are provided before predicting cocultures.")

        multiple_cocultures = True if len(self.coculture_files) > 1 else False
        self.predict_dir = time_based_dir(
            prefix="Prediction",
            base_path=self.output_dir,
            multiple_cocultures=multiple_cocultures
        )
        os.makedirs(self.predict_dir, exist_ok=True)

        for sample_file in self.coculture_files:

            sample_id = os.path.basename(sample_file)
            sample, _ = os.path.splitext(sample_id)

            _, data_df = fcsparser.parse(sample_file, reformat_meta=True)

            if 'Time' in data_df.columns:
                data_df = data_df.drop(columns=['Time'])

            if self.x_axis or self.y_axis or self.z_aixs not in data_df.columns:
                self.x_axis, self.y_axis, self.z_axis = data_df.columns[:3]

            # Get thresholds for uncertainty filtering
            if self.filter_out_uncertain and self.uncertainty_threshold is None:
                self.uncertainty_threshold = self.cs_uncertainty_threshold

            # Define common parameters
            predict_params = {
                "sample": sample,
                "model": self.model,
                "scaler": self.scaler,
                "label_encoder": self.le,
                "data_df": data_df,
                "predict_dir": self.predict_dir,
                "x_axis_combo": self.x_axis,
                "y_axis_combo": self.y_axis,
                "z_axis_combo": self.z_axis,
                "gating": self.gating,
                "scaling_constant": self.scaling_constant,
                "filter_out_uncertain": self.filter_out_uncertain,
                "uncertainty_threshold": self.uncertainty_threshold
            }
            # Add specific parameters based on gating
            if self.gating:
                predict_params.update({
                    "stain1": self.stain1_predict,
                    "stain2": self.stain2_predict,
                    "extra_stains": self.extra_stains
                })

            # Call predict function; df is the prediction dataframe after entropy filtering if step is applied
            predict(**predict_params)

        if multiple_cocultures:

            print("Merge prediction of all samples into a single file.")
            merge_prediction_results(self.predict_dir, "prediction")

            print("Merge prediction and uncertainties single file.")
            merge_prediction_results(self.predict_dir, "uncertainty")


def load_yaml(yaml_file: str):
    """
    Load a yaml file

    :param yaml_file: path to the YAML file
    """
    with open(yaml_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as exc:
            print(f"Error in configuration file: {exc}")
            sys.exit(1)


def parse_dicts(dir_list: [dict], entity: str, names: str=None):
    """
    Processes a list of directory info to extract file paths and optionally map them to names.

    Parameters:
    -----------
    dir_list : List containing directory info, each with 'path' and 'filenames'.
    entity : The entity being processed (e.g., "species_files").
    names : The key to map filenames to names (labels). If omitted, only file paths are returned.

    Returns:
    --------
    set or tuple
        - If `names` is provided, returns a tuple of:
            - A set of file paths.
            - A dictionary mapping names to file paths.
        - Otherwise, returns only the set of file paths.
    """
    all_files = set()
    if names:
        all_maps = {}

    for dir in dir_list:

        case_dir = dir["directory"]

        # Get pathway
        case_dir_path = case_dir["path"]
        if case_dir_path is None:
            raise f"Please specify path for the {entity} in the configuration file."
        if case_dir_path[0] == "~":
            case_dir_path = os.path.expanduser(case_dir_path)

        # Get filenames
        filenames = case_dir.get("filenames")
        if filenames is None:
            filenames = os.listdir(case_dir_path)
        files = [os.path.join(case_dir_path, x) for x in filenames]

        # Map filenames to labels
        if names:
            species_names = case_dir[names]
            if species_names is None:
                raise ValueError(f"""
                    Please provide {entity} in your configuration file.
                    Make sure you have as many filenames as {entity} and have the names
                    in the same order as their corresponding filenames so CellScanner can map them."""
                )

            # Make a list of the files mapping at each species name
            maps = defaultdict(list)

            for name, file in zip(species_names, files):
                maps[name].append(file)

            # Convert to regular dict (if needed)
            maps = dict(maps)

            # Update maps dictionary with maps of running directory
            all_maps.update(maps)

        # Update filenames set with filenames of running directory
        all_files.update(files)

    return (all_files, all_maps) if names else all_files


def get_param_value(param, conf):
    v = conf.get(param, {}).get("value") or conf.get(param, {}).get("name") or conf.get(param, {}).get("path")
    if v is None:
        v = conf.get(param).get("default")
    if v is None and param not in ["prev_trained_model"]:
        raise ValueError(f"Provide a value to the {param} parameter, or set back the default value based on the config.yml template.")
    return v


def get_extra_stains(conf):
    """
    Get extra stains provided by the user

    :return extra_stains (Dict): A dictionary with channel name as key and a set with the sign, threshold and label
    of the stain as value
    """
    extra_stains = {}
    extras = conf.get("extra_stains").get("stains")
    for stain in  extras:
        channel = stain.get("channel")
        sign = stain.get("sign")
        threshold = stain.get("value")
        label = stain.get("label")
        if all(x is not None for x in (channel, sign, threshold, label)):
            extra_stains[channel]  = (sign, threshold, label)
    return extra_stains


def get_stain_params(stain, conf):
    """
    Build a Stain instance based on the configuration file.
    """
    # Get params from the yaml file
    params = conf.get(stain)
    channel = params.get("channel")
    sign = params.get("sign")
    value = params.get("value")
    return build_stain(stain, channel, sign, value)


def build_stain(stain, channel, sign, value):
    # Check if all stain params are there
    if not all([sign, value]) and channel is not None:
        missing = [k for k, v in {"channel": channel, "sign": sign, "value": value}.items() if v is None]
        raise ValueError(f"Please provide {' and '.join(missing)} for {stain}.")

    elif channel is None:
        return Stain(channel=None, sign=None, value=None)

    return Stain(channel=channel, sign=sign, value=value)


# Main function to run CellScanner from the CLI
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CellScanner Command Line Interface")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to the configuration file (.yml)")

    # Parse the arguments
    args = parser.parse_args()

    cs = CellScannerCLI(args)

    if cs.model is None:
        cs.train_model()

    cs.predict_coculture()
