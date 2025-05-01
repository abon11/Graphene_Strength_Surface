import pandas as pd
import matplotlib.pyplot as plt
import local_config

def main():
    folder = f'{local_config.DATA_DIR}/defected_data'
    csv_file = f"{folder}/all_simulations.csv"
    
    # Define filters here
    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        "Defect Type": "SV",
        "Defect Percentage": 0.5
        # "Defect Random Seed": 1
    }

    range_filters = {
        # "Defect Percentage": (0.4, 0.6)
        "Defect Random Seed": (1, 42)
    }

    or_filters = {
        # "Defect Type": ["SV", "DV"]
    }

    color_by_field = "Defect Random Seed"
    show_pristine = True

    # Load, filter, and plot
    df = load_data(csv_file)

    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters)

    base_title = create_title(exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters)

    xdom = filtered_df[filtered_df["Strain Rate x"] >= filtered_df["Strain Rate y"]]
    ydom = filtered_df[filtered_df["Strain Rate y"] >= filtered_df["Strain Rate x"]]

    if show_pristine:
        pristine_df = get_pristine_subset(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters)
        pris_x = pristine_df[pristine_df["Strain Rate x"] >= pristine_df["Strain Rate y"]]
        pris_y = pristine_df[pristine_df["Strain Rate y"] >= pristine_df["Strain Rate x"]]
    else:
        pris_x = None
        pris_y = None

    plot_strengths(xdom, folder, f"{base_title}, Armchair", color_by_field, pristine_data=pris_x)
    plot_strengths(ydom, folder, f"{base_title}, Zigzag", color_by_field, pristine_data=pris_y)


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


def filter_data(df, exact_filters=None, range_filters=None, or_filters=None):
    if (exact_filters is None) and (range_filters is None) and (or_filters is None):
        print("Warning: plotting entire dataset!")

    """Filter the dataframe based on exact and range filters."""
    filtered_df = df.copy()

    # Exact matches
    if exact_filters:
        for column, value in exact_filters.items():
            filtered_df = filtered_df[filtered_df[column] == value]

    # Range matches
    if range_filters:
        for column, (min_val, max_val) in range_filters.items():
            filtered_df = filtered_df[
                (filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)
            ]

    # OR matches
    if or_filters:
        for column, allowed_values in or_filters.items():
            filtered_df = filtered_df[filtered_df[column].isin(allowed_values)]

    return filtered_df


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


def plot_strengths(df, folder, title, color_by_field, pristine_data=None):
    """Scatter plot of Strength_1 vs Strength_2."""
    if df.empty and (pristine_data is None or pristine_data.empty):
        print("No data matches the specified filters.")
        return
    
    colors, value_to_color = assign_colors(df, color_by_field=color_by_field)
    
    if colors is None:
        colors = ['blue'] * len(df)

    plt.figure(figsize=(8,6))
    plt.scatter(df["Strength_1"], df["Strength_2"], c=colors, alpha=0.7)
    plt.scatter(df["Strength_2"], df["Strength_1"], c=colors, alpha=0.7)
    if pristine_data is not None and not pristine_data.empty:
        plt.scatter(pristine_data["Strength_1"], pristine_data["Strength_2"], c='black', alpha=0.7, label='Pristine')
        plt.scatter(pristine_data["Strength_2"], pristine_data["Strength_1"], c='black', alpha=0.7)
    plt.xlabel(r'$\sigma_1$')
    plt.ylabel(r'$\sigma_2$')

    plt.plot([-50, 130], [0, 0], color='black')
    plt.plot([0, 0], [-50, 130], color='black')

    plt.xlim(-15, 130)
    plt.ylim(-15, 130)
    plt.title(title)

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
        plt.legend(handles, labels, title=legend_title, loc='best', frameon=True)

    fname = f"{folder}/plots/SS_{clean_title(title)}"
    plt.savefig(fname)
    plt.close()
    print(f"Plot saved to {fname}.")
   

if __name__ == "__main__":
    main()
