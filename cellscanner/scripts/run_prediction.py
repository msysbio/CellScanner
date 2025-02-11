import os
import math
import numpy as np
import pandas as pd
from typing import List

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

from .helpers import create_file_path, get_stains_from_panel, apply_gating, save_gating_results, compute_label_counts
from .illustrations import species_plot, uncertainty_plot, heterogeneity_pie_chart, heterogeneity_bar_plot, create_color_map

# Main function to be called from the worker
def predict(PredictionPanel=None, **kwargs):
    """
    Executes the prediction workflow, including loading models,
    predicting species, applying gating, performing heterogeneity analysis, and generating visualizations.

    :param PredictionPanel:
    :param kwargs:
    """
    gui = False

    if type(PredictionPanel).__name__ == "PredictionPanel":

        # Attempt to retrieve components from file_panel first
        model, scaler, label_encoder, scaling_constant = get_model_components(PredictionPanel.file_panel)

        # Fallback to train_panel if not found in file_panel
        if model is None:
            model, scaler, label_encoder, scaling_constant = get_model_components(PredictionPanel.train_panel)

        data_df = PredictionPanel.data_df
        output_dir = PredictionPanel.predict_dir

        x_axis_combo = PredictionPanel.x_axis_selector.combo.currentText()
        y_axis_combo = PredictionPanel.y_axis_selector.combo.currentText()
        z_axis_combo = PredictionPanel.z_axis_selector.combo.currentText()
        gating = PredictionPanel.gating_checkbox.isChecked()
        sample = PredictionPanel.sample

        # NOTE: In case a model has just been trained, then the GUI will update the uncertainty threshold
        # (PredictionPanel.uncertainty_threshold) accrordingly
        # Otherwise, a -1 value will be the default value for the uncertainty threshold, in which case,
        # the 0.5*max_entropy will be used as the threshold
        # If the user gives an uncertainty threshold, then it will be used as the threshold -- not as a quantile.
        filter_out_uncertain = PredictionPanel.uncertainty_filtering_checkbox.isChecked()
        uncertainty_threshold = float(PredictionPanel.uncertainty_threshold.value()) if filter_out_uncertain else None

        if gating:
            # Stains 1 and 2 for live/dead and cell/not cells
            stain1, stain2 = get_stains_from_panel(PredictionPanel)
            # Extra stains
            extra_stains = PredictionPanel.extra_stains
        gui = True

    else:
        sample = kwargs["sample"]
        model = kwargs["model"]
        scaler = kwargs["scaler"]
        label_encoder = kwargs["label_encoder"]
        data_df = kwargs["data_df"]
        output_dir = kwargs["predict_dir"]
        x_axis_combo = kwargs["x_axis_combo"]
        y_axis_combo = kwargs["y_axis_combo"]
        z_axis_combo = kwargs["z_axis_combo"]
        gating = kwargs["gating"]
        scaling_constant = kwargs["scaling_constant"]
        filter_out_uncertain = kwargs["filter_out_uncertain"]
        uncertainty_threshold = kwargs["uncertainty_threshold"]
        if gating:
            stain1, stain2 = kwargs["stain1"], kwargs["stain2"]
            extra_stains = kwargs["extra_stains"]

    # Get uncertainty threshold
    if filter_out_uncertain:
        number_of_classes = len(label_encoder.classes_)
        max_entropy = math.log(number_of_classes)
        if (uncertainty_threshold < 0 and uncertainty_threshold != -1.0) or uncertainty_threshold > max_entropy:
            raise ValueError("Uncertainty threshold must be between 0 and 1.")

        elif uncertainty_threshold == -1.0:
            uncertainty_threshold = 0.5 * max_entropy
            print(f"Threshold as 0.5 of max entropy: {uncertainty_threshold}, max entropy: {max_entropy}")

    # Predict the species in the coculture file
    predicted_classes, uncertainties, index_to_species = predict_species(
        data_df,
        model,
        scaler,
        label_encoder,
        scaling_constant
    )

    # Convert uncertainties to a Pandas Series
    data_df_pred = data_df.copy()

    # Drop the 'Time' column if it exists
    if 'Time' in data_df_pred.columns:
        data_df_pred = data_df_pred.drop(columns=['Time'])

    # Map prediction indices back to species names
    mapped_predictions = np.vectorize(index_to_species.get)(predicted_classes)

    # Add predictions and uncertainties to the coculture data
    data_df_pred['predictions'] = mapped_predictions

    # Ensure uncertainties is a Series with the same index
    uncertainties = pd.Series(uncertainties, index=data_df_pred.index)

    # Build df with predictions (species names) and uncertainties
    data_df_pred['uncertainties'] = uncertainties  # NOTE: This is the main df to work with

    # Filter out predictions of high entropy
    if filter_out_uncertain:
        data_df_pred.loc[data_df_pred["uncertainties"] > uncertainty_threshold, "predictions"] = "Unknown"

    # Gating -- may return a "state" column mentioning live - dead cells, it may not
    if gating:

        print("Run gating...")

        # Apply gating
        gating_df, all_labels = apply_gating(
            data_df_pred,
            stain1,
            stain2,
            extra_stains
        )
        # Save gating results
        save_gating_results(
            gating_df, output_dir, sample,
            x_axis_combo, y_axis_combo, z_axis_combo,
            all_labels
        )
        # Perform heterogeneity analysis
        hetero_df = gating_df.drop(columns=all_labels)

    else:
        hetero_df = data_df_pred.copy()

    # Calculate heterogeneity
    species_list = list(index_to_species.values())
    run_heterogeneity(hetero_df, species_list, output_dir, sample)

    # Save prediction results and plot the 3D scatter plot
    save_prediction_results(
        data_df_pred,
        species_list,
        output_dir,
        x_axis_combo, y_axis_combo, z_axis_combo,
        sample=sample,
        scaling_constant=scaling_constant,
        uncertainty_threshold=uncertainty_threshold,
        filter_out_uncertain=filter_out_uncertain,
        gating_df = gating_df if 'gating_df' in locals() else None,
        all_labels = all_labels if 'all_labels' in locals() else None
    )

    if not gui:
        return data_df_pred


# Functions to be used by the predict()
def predict_species(data_df, model, scaler, label_encoder, scaling_constant):
    """

    :param data_df:
    :param model:
    :param scaler:
    :param label_encoder:
    :param scaling_constant:

    :return predicted_classes (np.ndarray of ints): The predicted class indices
    :return uncertainties (np.ndarray of floats): The entropy values for each prediction
    :return index_to_species (dict): A mapping from class index to species name
    """

    # Select only numeric columns
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns

    # Apply arcsinh transformation only to numeric columns
    data_df_arcsinh = data_df.copy()
    data_df_arcsinh[numeric_cols] = np.arcsinh(data_df_arcsinh[numeric_cols] / scaling_constant)

    # Z-standardization
    X_co_scaled = scaler.transform(data_df_arcsinh[numeric_cols])

    # Predict the species
    predictions = model.predict(X_co_scaled)

    # Convert predictions to class labels
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate entropy for each prediction to represent uncertainty (using scipy.stats.entropy)
    uncertainties = entropy(predictions, axis=1)  # uncertainties example shape (50000,)

    # Create the index-to-species mapping dictionary
    index_to_species = {index: label for index, label in enumerate(label_encoder.classes_)}

    return predicted_classes, uncertainties, index_to_species


def save_prediction_results(
    data_df: pd.DataFrame, species_list: List,
    output_dir: str,
    x_axis, y_axis, z_axis,
    sample: str = None,
    scaling_constant: int = 150,
    uncertainty_threshold: float = 0.5,
    filter_out_uncertain: bool = False,
    gating_df: pd.DataFrame = None,
    all_labels: list = None
    ):
    """
    Saves prediction file for a coculture sample CellScanner prediction along with its corresponding species and uncertainty plolts.
    """
    # Ensure `data_df` is still a DataFrame and not an ndarray
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Expected a DataFrame for `data_df`, but got something else.")

    # Create filenames for the prediction counts CSV and html files
    outfile_predictions = create_file_path(output_dir, sample, 'raw_predictions', 'csv')
    outfile_prediction_counts = create_file_path(output_dir, sample, 'prediction_counts', 'csv')
    plot_path_species = create_file_path(output_dir, sample, '3D_coculture_predictions_species', 'html')

    # Save raw predictions and prediction counts to a CSV file
    data_df.to_csv(outfile_predictions)

    # Calculate and save uncertainty counts by species
    if filter_out_uncertain:
        uncertaint_dir = os.path.join(output_dir, "uncertainty_counts")
        os.makedirs(uncertaint_dir, exist_ok=True)
        outfile_uncertainties = create_file_path(uncertaint_dir, sample, 'uncertainty_counts', 'csv')
        plot_path_uncertainty = create_file_path(uncertaint_dir, sample, '3D_coculture_predictions_uncertainty', 'html')
        uncertainty_counts = data_df.groupby('predictions')['uncertainties'].agg(
            greater_than=lambda x: (x > uncertainty_threshold).sum(),
            less_than=lambda x: (x <= uncertainty_threshold).sum()
        )
        uncertainty_counts.to_csv(outfile_uncertainties)
        print("Uncertainty counts by species saved to:", outfile_uncertainties)

    # Save counts based on whether gating and/or uncertainty filtering has been applied
    output_data = {}  # output_file: data
    if filter_out_uncertain and gating_df is not None:

        print("Both gating and uncertainty filter..")
        df = gating_df[gating_df["uncertainties"] < uncertainty_threshold]
        output_data.update(compute_label_counts(gating_df, all_labels, output_dir, sample))

    elif gating_df is not None:
        print("Only gating..")
        output_data.update(compute_label_counts(gating_df, all_labels, output_dir, sample))

    elif filter_out_uncertain:
        print("Only uncertainty filter..")
        df = data_df[data_df["uncertainties"] < uncertainty_threshold]
        all_counts = df['predictions'].value_counts()
        output_data[outfile_prediction_counts] = all_counts

    else:
        print("No gating, no uncertainty filter..")
        all_counts = data_df['predictions'].value_counts()
        output_data[outfile_prediction_counts] = all_counts

    # Save sample's counts
    for outfile, df in output_data.items():
        df.to_csv(outfile)

    # ----------------
    # PLOTS
    # ----------------

    # Perform arcsinh transformation on numeric columns
    coculture_data_numeric = data_df.drop(columns=['predictions', 'uncertainties'])
    coculture_data_arcsin = np.arcsinh(coculture_data_numeric / scaling_constant)

    # Reintegrate 'predictions' and 'uncertainties' columns
    coculture_data_arcsin['predictions'] = data_df['predictions']

    # Dynamically generate color map based on the number of species
    color_map = create_color_map(species_list)

    # Plot 1: Species Plot
    species_plot(
        coc_arsin_df=coculture_data_arcsin,
        plot_path=plot_path_species,
        x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
        color_map=color_map
    )
    # Plot 2: Uncertainty Plot
    if filter_out_uncertain:
        coculture_data_arcsin['uncertainties'] = data_df['uncertainties']
        uncertainty_plot(
            coc_arcsin_df=coculture_data_arcsin,
            plot_path=plot_path_uncertainty,
            x_axis=x_axis, y_axis=y_axis, z_axis=z_axis
        )


def run_heterogeneity(df, species_list, output_dir, sample):
    """
    Calculate, plot and export to files heterogeneity metrics.

    :param df:
    :param species_list:
    :param output_dir:
    :param sample:
    """
    # Create a directory for heterogeneity results
    heterogeneity_dir = os.path.join(output_dir, 'heterogeneity_results')
    os.makedirs(heterogeneity_dir, exist_ok=True)

    hetero_df = df.select_dtypes(include='number')
    try:
        hetero_df.drop('uncertainties', axis=1, inplace=True)
    except:
        print("No uncertainties; dropped in gating_df.drop(columns=all_labels).")
        pass
    hetero_df['predictions'] = df['predictions']

    # Compute heterogeneity measures for the sample
    try:
        hetero1 = hetero_simple(hetero_df.iloc[:, :-1])
        hetero2 = hetero_mini_batch(hetero_df.iloc[:, :-1])
    except ValueError as e:
        raise ValueError("Error calculating heterogeneity.") from e

    # Compute heterogeneity metrics for each species
    hetero_results_list = []
    for species in species_list:

        df = hetero_df[hetero_df['predictions'] == species]

        # Calculate metrics
        try:
            hetero1 = hetero_simple(df.iloc[:, :-1])
            hetero2 = hetero_mini_batch(df.iloc[:, :-1], species)

            # # Build heterogeneity plots  --  decided not to plot
            # save_heterogeneity_plots(hetero1, hetero2, heterogeneity_dir, sample, species)

            # Append the result as a dictionary to the list
            hetero_results_list.append({
                "Species": species,
                "Simple Heterogeneity": hetero1,
                "Medoid Heterogeneity": hetero2
            })
        except ValueError as e:
            raise ValueError("Error calculating heterogeneity.") from e

    # Convert the list of results to a DataFrame
    hetero_results_df = pd.DataFrame(hetero_results_list)

    # Save the DataFrame to a CSV file
    res_file = "_".join([sample, "heterogeneity_results.csv"])
    hetero_res_file = os.path.join(heterogeneity_dir, res_file)
    hetero_results_df.to_csv(hetero_res_file, sep="\t", index=False)


def hetero_simple(data):
    """Calculate simple heterogeneity as the sum of mean ranges across all channels."""
    ranges = data.apply(np.ptp, axis=0)
    return np.sum(ranges.mean())


def hetero_mini_batch(data, species=None, type='av_diss'):
    """
    Uses a variant of K-Means clustering that is faster and more memory-efficient asking for a single cluster
    similar to computing the mean (or geometric center) of all points.
    The distance of points to this single centroid (central reference point) can be used to assess variability.

    It then used this to compute the distance between each pair data point from it.

    :param data:
    :param species:
    :param type:
    :return result: Maximum distance between the centroid and data points
    """
    # Use MiniBatchKMeans as an alternative
    if data.shape[0] == 0:
        if species == None:
            species = "global"
        print(f"No data for heterogeneity test for species {species}.")
        return np.nan
    # Use MiniBatchKMeans as an alternative
    try:
        kmeans = MiniBatchKMeans(n_clusters=1, batch_size=3080, n_init=3).fit(data)
    except ValueError:
        raise ValueError("MiniBatchKMeans failed to fit the data.")

    if type == 'diameter':
        # Uses np.max()
        result = np.max(pairwise_distances(kmeans.cluster_centers_[0].reshape(1, -1), data))
    elif type == 'av_diss':
        # Uses np.mean()
        result = np.mean(pairwise_distances(kmeans.cluster_centers_[0].reshape(1, -1), data))
    return result


def save_heterogeneity_plots(hetero1, hetero2, output_dir, sample, species = None):
    """
    Exports html heterogeneity pie chart and bar plot
    """
    # Values corresponding to each measure
    metrics_data = [hetero1, hetero2]
    labels = ['Simple Heterogeneity', 'Medoid Heterogeneity']
    colors = ['#ff9999', '#66b3ff']

    # plot dimensions
    plot_width = 800
    plot_height = 600

    # Pie chart
    heterogeneity_pie_chart(labels, metrics_data, colors, output_dir, sample, species, plot_width, plot_height)

    # Bar chart
    heterogeneity_bar_plot(labels, metrics_data, colors, output_dir, sample, species, plot_width, plot_height)


def get_model_components(panel):
    """
    Helper function to retrieve model, scaler, label encoder, and scaling constant from a panel.
    """
    model = getattr(panel, "model", None)
    scaler = getattr(panel, "scaler", None)
    label_encoder = getattr(panel, "le", None)
    scaling_constant = getattr(panel, "scaling_constant", None)
    scaling_constant_value = scaling_constant.spin_box.value() if scaling_constant else None
    return model, scaler, label_encoder, scaling_constant_value

