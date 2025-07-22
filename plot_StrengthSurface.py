import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import local_config
import plotly.express as px
import plotly.graph_objects as go

def main():
    folder = f'{local_config.DATA_DIR}/rotation_tests'
    # folder = f'{local_config.DATA_DIR}/defected_data'

    csv_file = f"{folder}/all_simulations.csv"
    
    # Define filters here
    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        "Defect Type": "None",
        # "Defect Percentage": 0.5,
        # # "Defect Random Seed": 1,
        # "Theta": 30
    }

    range_filters = {
        # "Defect Percentage": (0.4, 0.6)
        # "Defect Random Seed": (1, 1000)
        # "Theta": (0, 90)
        # "Simulation ID": (2575, 3000)
    }

    or_filters = {
        # "Defect Type": ["SV", "DV"]
        # "Theta": [0, 60]
    }

    color_by_field = "Theta"
    show_pristine = False

    # Load, filter, and plot
    df = load_data(csv_file)

    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters)
    filtered_df = duplicate_biaxial_rows(filtered_df)

    base_title = create_title(exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters)

    if show_pristine:
        pristine_df = get_pristine_subset(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters)
    else:
        pristine_df = None

    # plot_strengths(filtered_df, folder, f"{base_title}", color_by_field, pristine_data=pristine_df, legend=True)
    plot_strengths_3d(filtered_df, folder, f"{base_title}", color_by_field, pristine_data=pristine_df)


def load_data(csv_file):
    """Load the simulation data from CSV."""
    return pd.read_csv(csv_file)


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


def filter_data(df, exact_filters=None, range_filters=None, or_filters=None, dupe_thetas=False):
    """
    Filter df on exact, range, OR filters *and* make phantom copies
    of any isotropic points (Strain Rate x == Strain Rate y)
    at *every* Theta value the user asked for.
    """
    # Copy and pull out Theta filters
    filtered = df.copy()
    theta_exact = None
    theta_range = None
    theta_or = None

    if exact_filters and "Theta" in exact_filters:
        theta_exact = exact_filters["Theta"]
    if range_filters and "Theta" in range_filters:
        theta_range = range_filters["Theta"]
    if or_filters and "Theta" in or_filters:
        theta_or = or_filters["Theta"]

    # Build the clean dicts without Theta
    exact_clean = {k: v for k, v in (exact_filters or {}).items() if k != "Theta"}
    range_clean = {k: v for k, v in (range_filters or {}).items() if k != "Theta"}
    or_clean = {k: v for k, v in (or_filters    or {}).items() if k != "Theta"}

    # Apply non-Theta exact filters
    for col, val in exact_clean.items():
        if col == "Defect Type" and (val == "None" or val is None):
            # Match rows where the value is "None" or NaN
            mask = (filtered[col] == "None") | (filtered[col].isna())
            filtered = filtered[mask]
        else:
            filtered = filtered[filtered[col] == val]

    # Apply non-Theta range filters
    for col, (mn, mx) in range_clean.items():
        filtered = filtered[(filtered[col] >= mn) & (filtered[col] <= mx)]

    # Apply non-Theta OR filters
    for col, vals in or_clean.items():
        if col == "Defect Type":
            has_none = "None" in vals or None in vals
            val_set = set(vals) - {"None", None}

            mask = filtered[col].isin(val_set)
            if has_none:
                mask |= filtered[col].isna()
            filtered = filtered[mask]
        else:
            filtered = filtered[filtered[col].isin(vals)]

    # Separate iso vs aniso
    iso_mask = (filtered["Strain Rate x"] == filtered["Strain Rate y"]) & (filtered["Strain Rate xy"] == 0)
    aniso_mask = ~iso_mask

    # Build theta_list from whatever filters the user gave, but only from the Theta values present in the current filtered set.
    theta_list = []

    if theta_exact is not None:
        theta_list = [theta_exact]

    elif theta_or is not None:
        # only the exact ones they asked for (and that exist)
        theta_list = list(filtered.loc[filtered["Theta"].isin(theta_or), "Theta"].unique())

    elif theta_range is not None:
        mn, mx = theta_range
        # pull only the Theta values in [mn, mx] that are actually in filtered
        theta_list = list(filtered.loc[(filtered["Theta"] >= mn) & (filtered["Theta"] <= mx),"Theta"].unique())

    # Phantom-replicate iso rows
    iso_df = filtered[iso_mask]

    if dupe_thetas and theta_list:
        # Replicate isotropic rows to each requested theta
        phantoms = []
        for th in theta_list:
            tmp = iso_df.copy()
            tmp["Theta"] = th
            phantoms.append(tmp)
        iso_phantom_df = pd.concat(phantoms, ignore_index=True) if phantoms else pd.DataFrame(columns=filtered.columns)
    else:
        # Either don't duplicate or no theta filtering — keep original
        iso_phantom_df = iso_df.copy()

    # Now select aniso rows by Theta filters
    if theta_exact is not None:
        aniso_df = filtered[aniso_mask & (filtered["Theta"] == theta_exact)]
    elif theta_or is not None:
        aniso_df = filtered[aniso_mask & filtered["Theta"].isin(theta_or)]
    elif theta_range is not None:
        mn, mx = theta_range
        aniso_df = filtered[aniso_mask & (filtered["Theta"] >= mn) & (filtered["Theta"] <= mx)]
    else:
        # no Theta filtering at all
        aniso_df = filtered[aniso_mask]

    # Union iso_phantoms + aniso_df
    filtered = pd.concat([iso_phantom_df, aniso_df], ignore_index=True)
    result = flip_strengths(filtered)
    return result


def flip_strengths(df):
    # Create a flipped version of the DataFrame
    flipped = df.copy()
    flipped["Strength_1"], flipped["Strength_2"] = df["Strength_2"], df["Strength_1"]

    # Combine original and flipped
    df_doubled = pd.concat([df, flipped], ignore_index=True)
    return df_doubled


def duplicate_biaxial_rows(df):
    # Identify perfectly biaxial tension cases (ratio == 1.0 and maybe sigma_xy ≈ 0)
    is_biaxial = (df["Theta Requested"] == -1)

    # Extract the biaxial rows
    biaxial_rows = df[is_biaxial]

    # For each theta, duplicate the biaxial row and assign that theta
    new_rows = []
    for theta in range(0, 91, 10):
        if theta != -1:
            for _, row in biaxial_rows.iterrows():
                new_row = row.copy()
                new_row["Theta Requested"] = theta
                new_row["Theta"] = theta
                new_rows.append(new_row)

    df = df[df["Theta Requested"] != -1]  # remove the -1 so we don't plot it
    # Append duplicated rows to dataframe
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


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


def assign_colors(df, color_by_field=None):
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
    return colors, value_to_color


def plot_strengths(df, folder, title, color_by_field, pristine_data=None, legend=True):
    """Scatter plot of Strength_1 vs Strength_2."""
    if df.empty and (pristine_data is None or pristine_data.empty):
        print("No data matches the specified filters.")
        return
    
    colors, value_to_color = assign_colors(df, color_by_field=color_by_field)
    
    if colors is None:
        colors = ['blue'] * len(df)

    plt.figure(figsize=(8, 8))
    plt.scatter(df["Strength_1"], df["Strength_2"], c=colors, alpha=0.8, label='Defective')
    # plt.scatter(df["Strength_2"], df["Strength_1"], c=colors, alpha=0.8)
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
    # plt.title("Molecular Strength Surfaces of 0.5% SV Graphene", fontsize=20)

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
            plt.legend(handles, labels, title=legend_title, loc='best', frameon=True, fontsize=15)

    fname = f"{folder}/plots/SS_{clean_title(title)}"
    plt.tight_layout()
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
