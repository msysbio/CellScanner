import os
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy

from tensorflow.keras.models import load_model

import plotly.express as px
import plotly.graph_objects as go

from .helpers import get_abs_path


# Main function to be called from the worker
def predict(PredictionPanel=None, **kwargs):

    gui = False
    if type(PredictionPanel).__name__ == "PredictionPanel":

        # Attempt to retrieve components from file_panel first
        model, scaler, label_encoder, scaling_constant = get_model_components(PredictionPanel.file_panel)

        # Fallback to train_panel if not found in file_panel
        if model is None:
            model, scaler, label_encoder, scaling_constant = get_model_components(PredictionPanel.train_panel)

        data_df = PredictionPanel.data_df
        output_dir = PredictionPanel.predict_dir

        x_axis_combo = PredictionPanel.x_axis_combo.currentText()
        y_axis_combo = PredictionPanel.y_axis_combo.currentText()
        z_axis_combo = PredictionPanel.z_axis_combo.currentText()
        gating = PredictionPanel.gating_checkbox.isChecked()
        sample = PredictionPanel.sample

        if gating:
            # Stain 1
            stain1 = PredictionPanel.stain1_combo.currentText()  # It should be the column name
            stain1_relation = PredictionPanel.stain1_relation.currentText()
            stain1_threshold = float(PredictionPanel.stain1_threshold.text())
            # Stain 2
            stain2 = PredictionPanel.stain2_combo.currentText()  # It should be the column name
            stain2_relation = PredictionPanel.stain2_relation.currentText()
            stain2_threshold = float(PredictionPanel.stain2_threshold.text()) if PredictionPanel.stain2_threshold.text() else None
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
        if gating:
            Stain1 = kwargs["stain1"]
            stain1, stain1_relation, stain1_threshold = Stain1.channel, Stain1.sign, Stain1.value
            Stain2 = kwargs["stain2"]
            stain2, stain2_relation, stain2_threshold = Stain2.channel, Stain2.sign, Stain2.value

    # Predict the species in the coculture file
    predicted_classes, uncertainties,index_to_species, data_df = predict_species(
        data_df,
        model,
        scaler,
        label_encoder,
        scaling_constant
    )

    # Verify LabelEncoder classes
    print("LabelEncoder classes:", label_encoder.classes_)

    # Convert uncertainties to a Pandas Series
    # TODO: data_df_pred to be passed in the PredictionPanel
    data_df_pred = data_df.copy()
    uncertainties = pd.Series(uncertainties, index=data_df_pred.index)

    # Save prediction results and plot the 3D scatter plot
    save_prediction_results(predicted_classes, uncertainties,
                            data_df_pred, index_to_species, output_dir,
                            x_axis_combo, y_axis_combo, z_axis_combo, sample)

    # Gating
    if gating:
        # Apply gating
        data_df_pred = apply_gating(data_df_pred,
                                    stain1, stain1_relation, stain1_threshold,
                                    stain2, stain2_relation, stain2_threshold, scaling_constant)
        # Save gating results
        save_gating_results(data_df_pred, output_dir,
                            x_axis_combo, y_axis_combo, z_axis_combo)
        # Perform heterogeneity analysis
        filtered_data = data_df_pred[(data_df_pred['state'] == 'live') & (data_df_pred['predictions'] != 'background')]

        # Check and correct negative entries
        for col in filtered_data.columns[:-2]:
            min_val = filtered_data[col].min()
            if min_val < 0:
                filtered_data[col] = filtered_data[col] - min_val

        # Log transformation, handle zero values
        filtered_data.iloc[:, :-2] = filtered_data.iloc[:, :-2].replace(0, 1)
        filtered_data.iloc[:, :-2] = np.log(filtered_data.iloc[:, :-2])

        # Compute heterogeneity measures
        hetero1 = hetero_simple(filtered_data.iloc[:, :-2])
        hetero2 = hetero_mini_batch(filtered_data.iloc[:, :-2])

        # Create and save heterogeneity plots
        save_heterogeneity_plots(hetero1, hetero2, output_dir)

    if not gui:
        return data_df_pred


# Functions to be used by the predict()
def predict_species(data_df, model, scaler, label_encoder, scaling_constant):

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

    # Calculate entropy for each prediction to represent uncertainty
    uncertainties = entropy(predictions, axis=1)

    # Create the index-to-species mapping dictionary
    index_to_species = {index: label for index, label in enumerate(label_encoder.classes_)}

    return predicted_classes, uncertainties,index_to_species, data_df


def apply_gating(data_df, stain1, stain1_relation, stain1_threshold,
                 stain2=None, stain2_relation=None, stain2_threshold=None, scaling_constant=150):
    # Copy the DataFrame to not change the original data
    gated_data_df = data_df.copy()

    # Temporarily remove the 'predictions' column to avoid issues with numeric operations
    predictions_column = gated_data_df.pop('predictions') if 'predictions' in gated_data_df.columns else None

    # Apply arcsinh transformation with a cofactor
    cofactor = scaling_constant  # Cofactor

    for column in gated_data_df.select_dtypes(include=[np.number]).columns:
        gated_data_df[column] = np.arcsinh(gated_data_df[column] / cofactor)

    # Reintegrate the 'predictions' column after the arcsinh transformation
    if predictions_column is not None:
        gated_data_df['predictions'] = predictions_column

    # Initialize the 'state' column with 'live'
    gated_data_df['state'] = 'live'

    # Apply gating based on the first stain (live/dead)
    if stain1_relation == '>':
        gated_data_df.loc[gated_data_df[stain1] > stain1_threshold, 'state'] = 'inactive'
    elif stain1_relation == '<':
        gated_data_df.loc[gated_data_df[stain1] < stain1_threshold, 'state'] = 'inactive'

    # Apply gating based on the second stain (debris)
    if stain2 and stain2_threshold:
        if stain2_relation == '>':
            gated_data_df.loc[(gated_data_df['state'] == 'live') & (gated_data_df[stain2] > stain2_threshold), 'state'] = 'debris'
        elif stain2_relation == '<':
            gated_data_df.loc[(gated_data_df['state'] == 'live') & (gated_data_df[stain2] < stain2_threshold), 'state'] = 'debris'

    return gated_data_df


def save_prediction_results(predicted_classes, uncertainties, data_df, index_to_species, output_dir,
                            x_axis, y_axis, z_axis, sample=None, scaling_constant=150):
    # Ensure `data_df` is still a DataFrame and not an ndarray
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Expected a DataFrame for `data_df`, but got something else.")

    # Drop the 'Time' column if it exists
    if 'Time' in data_df.columns:
        data_df = data_df.drop(columns=['Time'])

    # Map prediction indices back to species names
    mapped_predictions = np.vectorize(index_to_species.get)(predicted_classes)

    # Add predictions and uncertainties to the coculture data
    data_df['predictions'] = mapped_predictions
    uncertainties = pd.Series(uncertainties, index=data_df.index)  # Ensure uncertainties is a Series with the same index
    data_df['uncertainties'] = uncertainties

    # Save the prediction counts to a CSV file
    prediction_counts = data_df['predictions'].value_counts()

    if sample is not None:
        outfile_predictions = os.path.join(output_dir, "_".join([sample, 'prediction_counts.csv']))
        outfile_uncertainties = os.path.join(output_dir, "_".join([sample, "uncertainty_counts.csv"]))
        plot_path_species = os.path.join(output_dir, "_".join([sample,'3D_coculture_predictions_species.html']))
        plot_path_uncertainty = os.path.join(output_dir, "_".join(['3D_coculture_predictions_uncertainty.html']))
    else:
        outfile_predictions = os.path.join(output_dir, 'prediction_counts.csv')
        outfile_uncertainties = os.path.join(output_dir, "uncertainty_counts")
        plot_path_species = os.path.join(output_dir, '3D_coculture_predictions_species.html')
        plot_path_uncertainty = os.path.join(output_dir, '3D_coculture_predictions_uncertainty.html')

    prediction_counts.to_csv(outfile_predictions)
    print("Prediction counts saved to:", outfile_predictions)

    # Calculate and save uncertainty counts by species
    uncertainty_counts = data_df.groupby('predictions')['uncertainties'].agg(
        greater_than_0_5=lambda x: (x > 0.5).sum(),
        less_than_equal_0_5=lambda x: (x <= 0.5).sum()
    )
    uncertainty_counts.to_csv(outfile_uncertainties)
    print("Uncertainty counts by species saved to:", outfile_uncertainties)

    # Perform arcsinh transformation on numeric columns
    coculture_data_numeric = data_df.drop(columns=['predictions', 'uncertainties'])
    coculture_data_arcsin = np.arcsinh(coculture_data_numeric / scaling_constant)

    # Reintegrate 'predictions' and 'uncertainties' columns
    coculture_data_arcsin['predictions'] = data_df['predictions']
    coculture_data_arcsin['uncertainties'] = data_df['uncertainties']

    # Dynamically generate color map based on the number of species
    species_list = list(index_to_species.values())
    color_map = {species: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, species in enumerate(species_list)}

    # Plot 1: Species Plot
    fig_species = px.scatter_3d(coculture_data_arcsin,
                                x=x_axis, y=y_axis, z=z_axis,
                                color='predictions',
                                color_discrete_map=color_map,
                                title="Coculture Predictions (Species)",
                                labels={x_axis: x_axis, y_axis: y_axis, z_axis: z_axis, 'predictions': 'Predictions'})

    # Adjust the layout for species plot
    fig_species.update_layout(width=1500, height=1000)
    fig_species.update_traces(marker=dict(size=1, opacity=0.8))

    # Save the species plot as an HTML file
    fig_species.write_html(plot_path_species)
    print("3D scatter plot (Species) saved to:", plot_path_species)

    # Plot 2: Uncertainty Plot
    fig_uncertainty = px.scatter_3d(
        coculture_data_arcsin,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        symbol='predictions',  # Different marker symbols for each species
        color='uncertainties',  # Use uncertainty for color scale
        color_continuous_scale='RdYlGn_r',  # Red for high uncertainty, green for low
        title="Coculture Predictions (Uncertainty)",
        hover_data={
            'uncertainties': True,  # Show uncertainties in hover info
            'predictions': True     # Show species in hover info
        },
        labels={x_axis: x_axis, y_axis: y_axis, z_axis: z_axis, 'uncertainties': 'Uncertainty/Entropy'}
    )

    # Adjust the layout for the uncertainty plot
    fig_uncertainty.update_layout(
        width=1500,
        height=1000,
        legend_title_text='Species',
        legend=dict(
            x=1.5,  # Move species legend further to the right of the plot
            y=0.5,  # Align vertically in the middle
            traceorder='normal',
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.6)',
        ),
        coloraxis_colorbar=dict(
            title='Uncertainty/Entropy',
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=300,
            yanchor="middle", y=0.5,
            xanchor="left", x=1.2  # Keep the color bar at the default position
        )
    )

    fig_uncertainty.update_traces(marker=dict(size=1, opacity=0.8))  # Adjust marker size and opacity

    # Save the uncertainty plot as an HTML file
    fig_uncertainty.write_html(plot_path_uncertainty)
    print("3D scatter plot (Uncertainty) saved to:", plot_path_uncertainty)


def save_gating_results(gated_data_df, output_dir, x_axis, y_axis, z_axis):
    # Create a directory for gating results
    gated_dir = os.path.join(output_dir, 'gated')
    os.makedirs(gated_dir, exist_ok=True)

    # Initialize an empty DataFrame to hold all state counts
    combined_counts_df = pd.DataFrame()

    # Iterate over each species and calculate the state counts
    species_names = gated_data_df['predictions'].unique()
    for species in species_names:
        state_counts = gated_data_df[gated_data_df['predictions'] == species]['state'].value_counts()
        state_counts.name = species  # Rename the series with the species name
        combined_counts_df = pd.concat([combined_counts_df, state_counts], axis=1)

    # Save the combined state counts to a single CSV file
    combined_counts_df.to_csv(os.path.join(gated_dir, 'combined_state_counts.csv'))

    # 3D plot creation for gated data
    fig = go.Figure()

    # Unique states and predictions for color
    states = gated_data_df['state'].unique()

    # Color map
    state_colors = {'live': 'skyblue', 'inactive': 'firebrick', 'debris': 'darkslategrey'}

    # Plot each combination of state and prediction
    for state in states:
        for species in species_names:
            df_filtered = gated_data_df[(gated_data_df['state'] == state) & (gated_data_df['predictions'] == species)]
            fig.add_trace(go.Scatter3d(
                x=df_filtered[x_axis],
                y=df_filtered[y_axis],
                z=df_filtered[z_axis],
                mode='markers',
                marker=dict(
                    size=1,
                    symbol='circle',  # Markers for predictions
                    color=state_colors[state],  # Color by state
                ),
                name=f'{state} - {species}'
            ))

    # Layout adjustments
    fig.update_layout(
        title="3D Scatter Plot of Gated Data by State and Prediction",
        scene=dict(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            zaxis_title=z_axis
        )
    )
    fig.update_layout(width=1000, height=800)

    # Adjusting the legend size
    fig.update_layout(
        legend=dict(
            title_font_size=20,
            font=dict(
                size=17,
            ),
        )
    )

    # Save the gated 3D plot as an HTML file
    plot_path = os.path.join(gated_dir, '3D_Gating_predictions_coculture.html')
    fig.write_html(plot_path)
    print("3D scatter plot for gated data saved to:", plot_path)


def hetero_simple(data):
    # Calculate simple heterogeneity as the sum of mean ranges across all channels.
    ranges = data.apply(np.ptp, axis=0)
    return np.sum(ranges.mean())


def hetero_mini_batch(data, type='av_diss'):
    # Use MiniBatchKMeans as an alternative
    kmeans = MiniBatchKMeans(n_clusters=1, batch_size=3080, n_init=3).fit(data)
    if type == 'diameter':
        result = np.max(pairwise_distances(kmeans.cluster_centers_[0].reshape(1, -1), data))
    elif type == 'av_diss':
        result = np.mean(pairwise_distances(kmeans.cluster_centers_[0].reshape(1, -1), data))
    return result


def save_heterogeneity_plots(hetero1, hetero2, output_dir):

    # Create a directory for heterogeneity results
    heterogeneity_dir = os.path.join(output_dir, 'heterogeneity_results')
    os.makedirs(heterogeneity_dir, exist_ok=True)

    # Values corresponding to each measure
    sizes = [hetero1, hetero2]
    labels = ['Simple Heterogeneity', 'Medoid Heterogeneity']
    colors = ['#ff9999', '#66b3ff']

    # plot dimensions
    plot_width = 800
    plot_height = 600

    # Pie chart
    fig1 = go.Figure(data=[go.Pie(labels=labels, values=sizes, marker_colors=colors, hole=.3)])
    fig1.update_layout(title_text='Heterogeneity of the Sample',width=plot_width,
        height=plot_height)
    pie_chart_path = os.path.join(heterogeneity_dir, 'heterogeneity_pie_chart.html')
    fig1.write_html(pie_chart_path)
    print(f"Pie chart saved to: {pie_chart_path}")

    # Bar chart
    fig2 = go.Figure(data=[go.Bar(x=labels, y=sizes, marker_color=['blue', 'green'])])
    fig2.update_layout(
        title='Comparison of Heterogeneity Measures',
        xaxis_title='Heterogeneity Measure',
        yaxis_title='Value',
        xaxis_tickangle=-45,
        width=plot_width,
        height=plot_height)

    bar_chart_path = os.path.join(heterogeneity_dir, 'heterogeneity_bar_chart.html')
    fig2.write_html(bar_chart_path)
    print(f"Bar chart saved to: {bar_chart_path}")


def merge_prediction_results(output_dir, prediction_type):

    if prediction_type not in ["prediction", "uncertainty"]:
        raise ValueError(f"Please provide a valide prediction_type: 'prediction|uncertainty'")

    dfs = []

    pattern = "_".join([prediction_type, "counts"])

    # Loop through all files in the directory
    for file_name in os.listdir(output_dir):
        if pattern not in file_name:
            continue
        file_path = os.path.join(output_dir, file_name)
        # Read each file as a DataFrame
        df = pd.read_csv(file_path, sep=",")  # Adjust separator if needed
        # Rename the "count" column to the filename (without extension)
        new_column_name = file_name.split(pattern)[0][:-1]
        df = df.rename(columns={"count": new_column_name})
        dfs.append(df)

    # Merge all DataFrames on the "predictions" column
    result = pd.concat(dfs, axis=1).loc[:,~pd.concat(dfs, axis=1).columns.duplicated()]

    # Save the final result to a CSV file
    merged_filename = "".join(["merged_", pattern, ".csv"])
    merged_file = os.path.join(output_dir, merged_filename)
    result.to_csv(merged_file, index=False)


def get_model_components(panel):
    """
    Helper function to retrieve model, scaler, label encoder, and scaling constant from a panel.
    """
    model = getattr(panel, "model", None)
    scaler = getattr(panel, "scaler", None)
    label_encoder = getattr(panel, "le", None)
    scaling_constant = getattr(panel, "scaling_constant", None)
    scaling_constant_value = scaling_constant.value() if scaling_constant else None
    return model, scaler, label_encoder, scaling_constant_value
