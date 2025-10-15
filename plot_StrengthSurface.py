"""
This plots a strength surface from the filters. Can do 2D or 3D
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import local_config
import plotly.express as px
import plotly.graph_objects as go
from filter_csv import filter_data

def main():
    # ========== USER INTERFACE ==========
    folder = f'{local_config.DATA_DIR}/rotation_tests'
    # folder = f'{local_config.DATA_DIR}/defected_data'

    csv_file = f"{folder}/all_simulations.csv"
    
    # Define filters here
    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        "Defects": '{"SV": 0.25, "DV": 0.25}',
        # "Defects": "None",
        # "Defect Random Seed": 67,
        "Theta Requested": 0,
        # "Strain Rate x": -0.00005,
    }

    range_filters = {
        # "Defect Percentage": (0.4, 0.6)
        # "Defect Random Seed": (0, 100)
        # "Theta": (0, 30)
        # "Simulation ID": (2575, 3000)
    }

    or_filters = {
        # "Defects": ['{"SV": 0.5}', '{"DV": 0.5}', '{"SV": 0.25, "DV": 0.25}'],
        # "Defect Random Seed": [0, 90]
        # "Theta Requested": [0, 90]
        # "Strain Rate x": [-0.00005, -0.00006]
    }
    # ====================================
    color_by_field = "Defects"
    show_pristine = False

    # Load, filter, and plot
    df = pd.read_csv(csv_file)

    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, 
                              flip_strengths=True, remove_biaxial=False, remove_dupes=True, duplic_freq=(0, 91, 10),
                              only_uniaxial=False)

    base_title = create_title(exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters)

    if show_pristine:
        pristine_df = get_pristine_subset(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters)
    else:
        pristine_df = None

    plot_strengths(filtered_df, folder, f"{base_title}TESTTTTT", color_by_field, pristine_data=pristine_df, legend=True, only_show=True)
    # plot_strengths_3d(filtered_df, folder, f"{base_title}", color_by_field, pristine_data=pristine_df)


def get_pristine_subset(df, exact_filters=None, range_filters=None, or_filters=None):
    """Return only the pristine rows that match all filters *except* defect fields."""
    pristine_df = df[
        (df["Defect Type"].isna() | (df["Defect Percentage"] == "None")) & 
        ((df["Defect Percentage"].isna()) | (df["Defect Percentage"] == 0.0))
    ]

    exclude_keys = ["Defect Type", "Defect Percentage", "Defect Random Seed"]

    exact_clean = {k: v for k, v in (exact_filters or {}).items() if k not in exclude_keys}
    range_clean = {k: v for k, v in (range_filters or {}).items() if k not in exclude_keys}
    or_clean    = {k: v for k, v in (or_filters or {}).items() if k not in exclude_keys}

    return filter_data(pristine_df, exact_filters=exact_clean, range_filters=range_clean, or_filters=or_clean)


def extract_field_string(field_name, exact_filters=None, range_filters=None, or_filters=None, suffix=""):
    """Extract a string representation of a field based on available filters."""
    # Check for exact match
    if exact_filters and field_name in exact_filters:
        return f"{exact_filters[field_name]}{suffix}"

    # Check for OR match
    if or_filters and field_name in or_filters:
        values = or_filters[field_name]
        return f"{' or '.join(str(v) for v in values)}{suffix}"

    # Check for range match
    if range_filters and field_name in range_filters:
        min_val, max_val = range_filters[field_name]
        if min_val == max_val:
            return f"{min_val}{suffix}"
        else:
            return f"{min_val}-{max_val}{suffix}"

    return None


def create_title(exact_filters=None, range_filters=None, or_filters=None):
    """Create a dynamic plot title based on filtering parameters."""
    title_parts = []

    # Handle atom dimensions manually (because it combines two fields)
    if exact_filters:
        num_x = exact_filters.get("Num Atoms x")
        num_y = exact_filters.get("Num Atoms y")
        if num_x and num_y:
            title_parts.append(f"{num_x}x{num_y}")

    # Handle Defect Type
    defect_type_str = extract_field_string("Defect Type", exact_filters, range_filters, or_filters)
    if defect_type_str:
        title_parts.append(defect_type_str)

    # Handle Defect Percentage
    defect_pct_str = extract_field_string("Defect Percentage", exact_filters, range_filters, or_filters, suffix="%")
    if defect_pct_str:
        title_parts.append(defect_pct_str)

    # Handle Defect Random Seed
    defect_rs_str = extract_field_string("Defect Random Seed", exact_filters, range_filters, or_filters)
    if defect_rs_str:
        title_parts.append(defect_rs_str)

    # Handle Theta
    theta_str = extract_field_string("Theta", exact_filters, range_filters, or_filters, suffix="deg")
    if theta_str:
        title_parts.append(theta_str)

    # Handle Theta Requested
    thetareq_str = extract_field_string("Theta Requested", exact_filters, range_filters, or_filters, suffix="deg")
    if thetareq_str:
        title_parts.append(thetareq_str)

    # Join all parts
    return ", ".join(title_parts)


def clean_title(title):
    """Convert a plot title into a filename-safe string."""
    # Replace spaces with underscores
    title = title.replace(" ", "_")
    # Remove commas
    title = title.replace(",", "")
    # Replace percentage sign with pct
    title = title.replace("%", "pct")
    # Replace periods with nothing (or underscore if you prefer)
    title = title.replace(".", "-")
    return title


def assign_colors(df, color_by_field=None, override=True):
    """Assign a color to each point based on the given field."""
    if color_by_field is None or color_by_field not in df.columns:
        return ['blue'] * len(df), {}
    
    unique_values = sorted(df[color_by_field].dropna().unique())
    n_colors = len(unique_values)

    if n_colors <= 9:
        colormap = plt.colormaps.get_cmap('Set1')  # Strong separation
    else:
        colormap = plt.colormaps.get_cmap('tab20')  # fallback for many categories

    value_to_color = {val: colormap(i / max(n_colors - 1, 1)) for i, val in enumerate(unique_values)}
    colors = [value_to_color[val] for val in df[color_by_field]]

    if override:
        # Define the manual mapping here
        manual_colors = ['blue', 'red', 'green']
        value_to_color = {val: manual_colors[i % len(manual_colors)] for i, val in enumerate(unique_values)}
        colors = [value_to_color[val] for val in df[color_by_field]]
        return colors, value_to_color

    return colors, value_to_color


def plot_strengths(df, folder, title, color_by_field, pristine_data=None, legend=True, only_show=False):
    """Scatter plot of Strength_1 vs Strength_2."""
    if df.empty and (pristine_data is None or pristine_data.empty):
        print("No data matches the specified filters.")
        return
    
    colors, value_to_color = assign_colors(df, color_by_field=color_by_field)
    
    if colors is None:
        colors = ['blue'] * len(df)

    plt.figure(figsize=(8, 8))
    # plt.scatter(df["Strength_1"], df["Strength_2"], c=df["Theta"], alpha=0.7, label='Defective')
    # plt.colorbar(label="Theta")
    plt.scatter(df["Strength_2"], df["Strength_1"], c=colors, alpha=0.2)
    if pristine_data is not None and not pristine_data.empty:
        plt.scatter(pristine_data["Strength_1"], pristine_data["Strength_2"], c='black', alpha=0.7, label='Pristine')
        # plt.scatter(pristine_data["Strength_2"], pristine_data["Strength_1"], c='black', alpha=0.7)
    plt.xlabel(r'$\sigma_1$ (GPa)', fontsize=18)
    plt.ylabel(r'$\sigma_2$ (GPa)', fontsize=18)

    plt.plot([-50, 130], [0, 0], color='black')
    plt.plot([0, 0], [-50, 130], color='black')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlim(-15, 130)
    plt.ylim(-15, 130)
    plt.title(title, fontsize=20)
    plt.title("MD Strength Surface - Armchair Mixed Defects", fontsize=20)

    # plt.legend(fontsize=15)

    # create legend
    if value_to_color:
        handles = []
        labels = []
        for val, color in value_to_color.items():
            # Format value cleanly: no decimal if not needed
            if isinstance(val, float) and val.is_integer():
                val_str = str(int(val))
            else:
                val_str = str(val)
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
            labels.append(val_str)

        # Add pristine legend entry
        if pristine_data is not None and not pristine_data.empty:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8))
            labels.append("Pristine")

        legend_title = color_by_field if color_by_field else "Legend"
        if legend:
            plt.legend(handles, labels, title=legend_title, loc='best', frameon=True, fontsize=15, title_fontsize=17)

    fname = f"{folder}/plots/SS_{clean_title(title)}"
    plt.tight_layout()
    if only_show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()
        print(f"Plot saved to {fname}.")


def plot_strengths_3d(df, folder, title, color_by_field, pristine_data=None):
    fig = px.scatter_3d(
        df,
        x="Strength_1",
        y="Strength_2",
        z="Theta",
        color=color_by_field if color_by_field in df.columns else None,
        title="Strength Surface by Dominant Principal Direction Angle",
        labels={"Theta": "Angle (deg)", "Strength_1": "σ₁", "Strength_2": "σ₂"},
        opacity=0.8,
        color_continuous_scale="Bluered",
    )

    fig.update_layout(
        scene=dict(
            xaxis_title_font=dict(size=35),
            yaxis_title_font=dict(size=35),
            zaxis_title_font=dict(size=35),
            xaxis=dict(tickfont=dict(size=18)),
            yaxis=dict(tickfont=dict(size=18)),
            zaxis=dict(tickfont=dict(size=18)),
        ),
        title_font=dict(size=40)
    )

    fig.update_coloraxes(colorbar_title_font=dict(size=35), colorbar_tickfont=dict(size=26))


    # write out a self‐contained HTML you can open in any browser
    html_path = f"{folder}/plots/3D_SS_{clean_title(title)}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Interactive 3D plot saved to {html_path}")

if __name__ == "__main__":
    main()
