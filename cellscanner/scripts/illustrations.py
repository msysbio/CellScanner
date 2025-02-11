import os
import plotly.express as px
import plotly.graph_objects as go


def create_color_map(alist):
    # Dynamically generate color map based on the number of species
    color_map = {entity: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)] for i, entity in enumerate(alist)}
    return color_map

def create_shape_map(alist):
    shapes = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up',
              'triangle-down', 'triangle-left', 'triangle-right', 'pentagon',
              'hexagon', 'hexagon2', 'octagon', 'star', 'star-triangle-up',
              'star-triangle-down', 'star-square', 'star-diamond',
              'diamond-tall', 'diamond-wide']
    if len(alist) > 20:
        raise ValueError(f"CellScanner can handle plots with up to 20 classes. A case with {len(alist)} was given.")
    shape_map = {entity: shapes[i % len(shapes)] for i, entity in enumerate(alist)}
    return shape_map

def umap_plot(cleaned_data, embedding, model_dir, title, indices_to_keep=None):
    # Generate UMAP plot after filtering
    fig = px.scatter_3d(
        x=embedding[:, 0] if indices_to_keep is None else embedding[indices_to_keep, 0],
        y=embedding[:, 1] if indices_to_keep is None else embedding[indices_to_keep, 1],
        z=embedding[:, 2] if indices_to_keep is None else embedding[indices_to_keep, 2],
        color=cleaned_data['Species'],
        title=f'UMAP Projection {title} Filtering',
        opacity=0.7,
        labels={
            'x': 'UMAP1',
            'y': 'UMAP2',
            'z': 'UMAP3',
            'color': 'Species'
        },
    )
    fig.update_traces(marker=dict(size=5))
    html_file = f'umap_{title}_filtering.html'
    umap_after_path = os.path.join(model_dir, html_file)
    fig.write_html(umap_after_path)


def species_plot(coc_arsin_df, x_axis, y_axis, z_axis, color_map, plot_path):

    fig_species = px.scatter_3d(coc_arsin_df,
                                x=x_axis, y=y_axis, z=z_axis,
                                color='predictions',
                                color_discrete_map=color_map,
                                title="Coculture Predictions (Species)",
                                labels={x_axis: x_axis, y_axis: y_axis, z_axis: z_axis, 'predictions': 'Predictions'})

    # Adjust the layout for species plot
    fig_species.update_layout(width=1500, height=1000)
    fig_species.update_traces(marker=dict(size=5, opacity=0.8))

    # Save the species plot as an HTML file
    fig_species.write_html(plot_path)


def uncertainty_plot(coc_arcsin_df, x_axis, y_axis, z_axis, plot_path):

    fig_uncertainty = px.scatter_3d(
        coc_arcsin_df,
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

    fig_uncertainty.update_traces(marker=dict(size=5, opacity=0.8))  # Adjust marker size and opacity

    # Save the uncertainty plot as an HTML file
    fig_uncertainty.write_html(plot_path)
    print("3D scatter plot (Uncertainty) saved to:", plot_path)


def heterogeneity_pie_chart(labels, metrics_data, colors, heterogeneity_dir, sample, species=None, plot_width=800, plot_height=600):

    fig1 = go.Figure(data=[go.Pie(labels=labels, values=metrics_data, marker_colors=colors, hole=.3)])

    fig1.update_layout(
        title_text='Heterogeneity of the Sample',
        width=plot_width,
        height=plot_height
    )
    pie_chart_path = os.path.join(
        heterogeneity_dir, f"{sample}_{species}_heterogeneity_pie_chart" if species else f"{sample}_heterogeneity_pie_chart"
    )
    fig1.write_html(pie_chart_path)


def heterogeneity_bar_plot(labels, metrics_data, colors, heterogeneity_dir, sample, species=None, plot_width=800, plot_height=600):

    fig2 = go.Figure(data=[go.Bar(x=labels, y=metrics_data, marker_color=colors)])
    fig2.update_layout(
        title='Comparison of Heterogeneity Measures',
        xaxis_title='Heterogeneity Measure',
        yaxis_title='Value',
        xaxis_tickangle=-45,
        width=plot_width,
        height=plot_height
    )
    bar_chart_path = os.path.join(
        heterogeneity_dir, f"{sample}_{species}_heterogeneity_bar_chart.html" if species else f"{sample}_heterogeneity_bar_chart.html"
    )
    fig2.write_html(bar_chart_path)


def gating_plot(gated_data_df, species_names, x_axis, y_axis, z_axis, gated_dir, sample, all_labels):


    for label in all_labels:

        # 3D plot creation for gated data
        fig = go.Figure()

        # Unique states and predictions for color
        states = gated_data_df[label].unique()

        # Color map
        state_colors = create_color_map(states)
        shape_map = create_shape_map(species_names)

        # Plot each combination of state and prediction
        for state in states:
            for species in species_names:
                df_filtered = gated_data_df[(gated_data_df[label] == state) & (gated_data_df['predictions'] == species)]
                fig.add_trace(go.Scatter3d(
                    x=df_filtered[x_axis],
                    y=df_filtered[y_axis],
                    z=df_filtered[z_axis],
                    mode='markers',
                    marker=dict(
                        size=5,
                        symbol=shape_map[species],  # Markers for predictions
                        color=state_colors[state],  # Color by state
                    ),
                    name=f'{state} - {species}'
                ))
        # Layout adjustments
        fig.update_layout(
            title=f"3D Scatter Plot of Gated Data by State and Prediction for {label}",
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
        plot_path = os.path.join(
            gated_dir, "_".join([
                sample,
                "_".join([label, '3D_Gating_predictions_coculture.html'])
            ])
        )
        fig.write_html(plot_path)
