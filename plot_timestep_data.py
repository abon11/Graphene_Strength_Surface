""" 
This plots the detailed timestep data. It is sort of working with the filters, but not great.
The function plot_many_detailed() is the best function in terms of actually working seamlessly
This script sucks but it sorta works lol... will revisit if i need it again
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import local_config
from filter_csv import filter_data


def main():
    # ========== USER INTERFACE ==========

    x_column = "Strain_1"
    y_column = "PrincipalStress_1"
    label_col = "Strain Rate x"

    folder = f'{local_config.DATA_DIR}/rotation_tests'
    csv_file = f"{folder}/all_simulations.csv"

    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        # "Defects": "{\"DV\": 0.25, \"SV\": 0.25}",  # will match NaN or "None"
        # "Defect Random Seed": 0,
        # "Theta Requested": 0,
        # "Strain Rate x": 0.001,
        # "Strain Rate y": 0.0
    }

    range_filters = {
        # "Defect Percentage": (0.4, 0.6),
        # "Simulation ID": (2670, 2913)
        # "Defect Random Seed": (0, 19)
    }

    or_filters = {
        # "Defects": ["{\"DV\": 0.25, \"SV\": 0.25}", "{\"DV\": 0.5}", "{\"SV\": 0.5}"],
        "Simulation ID": [2926, 36907]
    }
    # ====================================
    df = pd.read_csv(csv_file)
    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, only_uniaxial=True)
   
    # plot_detailed_data([f'{folder}/sim00400/dump.csv'], x_column, y_column, all_sims, label_col, output_file=f"{folder}combined_StressStrain.png")

    # plot_allsims_data(all_sims, list(range(2663, 2681)), 'Strain Rate x', 'Strength_1', output_file=f"{folder}strength_vs_StrainRate.png")
    # if exact_filters["Theta Requested"] == 0:
    #     orient = 'Armchair'
    # elif exact_filters["Theta Requested"] == 90:
    #     orient = 'Zigzag'
    # else:
    #     orient = None

    plot_many_detailed(filtered_df, x_column, y_column, folder, title="Pristine", labelby="Simulation ID")


def plot_many_detailed(df, x_col, y_col, folder, color=None, label_prefix="sim", title=None, labelby=None):
    """
    For each row in `df`, loads simulation results from {folder}/sim{SIMID}/sim{SIMID}.csv,
    and plots (x_col, y_col) on the same matplotlib Axes.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'Simulation ID' column.
        x_col (str): Column name to use for x-axis from sim CSV.
        y_col (str): Column name to use for y-axis from sim CSV.
        folder (str): Root folder containing simulation subfolders.
        color (str or list, optional): Color or list of colors to cycle through.
        label_prefix (str): Prefix for line labels in the legend.
    """

    # avg_strain_sv = filter_data(df, exact_filters={"Defects": "{\"SV\": 0.5}"}, shift_theta=False)["CritStrain_1"].mean()
    # avg_strain_mx = filter_data(df, exact_filters={"Defects": "{\"DV\": 0.25, \"SV\": 0.25}"}, shift_theta=False)["CritStrain_1"].mean()
    # avg_strain_dv = filter_data(df, exact_filters={"Defects": "{\"DV\": 0.5}"}, shift_theta=False)["CritStrain_1"].mean()

    # avg_strength_sv = filter_data(df, exact_filters={"Defects": "{\"SV\": 0.5}"}, shift_theta=False)["Strength_1"].mean()
    # avg_strength_mx = filter_data(df, exact_filters={"Defects": "{\"DV\": 0.25, \"SV\": 0.25}"}, shift_theta=False)["Strength_1"].mean()
    # avg_strength_dv = filter_data(df, exact_filters={"Defects": "{\"DV\": 0.5}"}, shift_theta=False)["Strength_1"].mean()

    potential_colors=['red', 'blue', 'green', 'purple', 'pink', 'orange', 'grey', 'gold', 'brown']
    color_dict = {}

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, row in df.iterrows():
        sim_id = str(row["Simulation ID"]).zfill(5)
        sim_path = os.path.join(folder, f"sim{sim_id}", f"sim{sim_id}.csv")

        if not os.path.isfile(sim_path):
            print(f"[Warning] File not found: {sim_path}")
            continue

        try:
            sim_df = pd.read_csv(sim_path)

            # Determine label/color
            if labelby is not None and labelby in df.columns:
                label_val = str(row[labelby])
                lab = f"{labelby}: {label_val}"
                if label_val not in color_dict:
                    # color_dict[str(label_val)] = potential_colors[len(color_dict) % len(potential_colors)]
                    # hardcoding this in so the colors are consistent across plots
                    if label_val == '{"SV": 0.5}':
                        color_dict[label_val] = 'red'
                    elif label_val == '{"DV": 0.5}':
                        color_dict[label_val] = 'blue'
                    elif label_val == '{"DV": 0.25,"SV": 0.25}':
                        color_dict[label_val] = 'green'
                    else:
                        color_dict[label_val] = potential_colors[len(color_dict) % len(potential_colors)]
                if int(label_val) == 2926:
                    lab = "Armchair"
                else:
                    lab = "Zigzag"
            else:
                lab = f"{label_prefix}{sim_id}"

            _, current_labels = ax.get_legend_handles_labels()
            ax.plot(
                sim_df[x_col],
                sim_df[y_col],
                label=lab if lab not in current_labels else None,
                color=color_dict[label_val] if label_val is not None else color[idx % len(color)] if color is not None else None,
                alpha=0.8
            )
        except Exception as e:
            print(f"[Error] Failed to plot sim{sim_id}: {e}")

    # ax.plot([avg_strain_sv, avg_strain_sv], [0, avg_strength_sv], color='red', linewidth=3, linestyle='-', alpha=0.8)
    # ax.plot([avg_strain_mx, avg_strain_mx], [0, avg_strength_mx], color='green', linewidth=3, linestyle='-', alpha=0.8)
    # ax.plot([avg_strain_dv, avg_strain_dv], [0, avg_strength_dv], color='blue', linewidth=3, linestyle='--', alpha=0.8)
    
    ax.set_xlabel("Strain", fontsize=18)
    ax.set_ylabel("Stress (GPa)", fontsize=18)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if title is not None:
        ax.set_title(f"Stress vs strain: {title}", fontsize=20)
    else: 
        ax.set_title("Stress vs strain", fontsize=20)

    ax.legend(fontsize=12)
    fig.tight_layout()
    if title is not None:
        filename = f"{folder}/plots/sig_eps_{title}.png"
    else:
        filename = f"{folder}/plots/stress_strain.png"
    fig.savefig(filename)
    print(f"Plot saved to {filename}")
    


def plot_detailed_data(csv_files, x_column, y_column, lookup_file, label_column, output_file=None):
    """
    Plots specified columns from multiple CSV files on the same graph, with labels based on a lookup CSV.

    Args:
        csv_files (list of str): List of paths to CSV files.
        x_column (str): The name of the column to use for the x-axis.
        y_column (str): The name of the column to use for the y-axis.
        lookup_file (str): Path to the CSV file containing metadata for simulations.
        output_file (str, optional): Path to save the plot. If None, displays the plot.
    """
    # Load the lookup file (all_simulations)
    lookup_df = pd.read_csv(lookup_file)
    lookup_df["Simulation ID"] = lookup_df["Simulation ID"].astype(str).str.zfill(5)  # Ensure IDs are 5 digits

    # Check that the label column exists
    if label_column not in lookup_df.columns:
        raise ValueError(f"Column '{label_column}' not found in {lookup_file}")

    plt.figure(figsize=(10, 6))

    for csv_file in csv_files:
        # Ensure the file exists
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue

        # Extract the Simulation ID from the filename
        sim_id = os.path.basename(csv_file).split(".")[0].replace("sim", "")
        
        # Find the corresponding Target Pressure x (bar)
        match = lookup_df[lookup_df["Simulation ID"] == sim_id]
        if match.empty:
            print(f"Simulation ID {sim_id} not found in {lookup_file}")
            label = f"sim{sim_id}"  # Fallback to the sim ID
        else:
            label = f"{label_column}: {match[label_column].iloc[0]}"

        # Load the simulation CSV
        df = pd.read_csv(csv_file)

        # Ensure the specified columns exist
        if x_column not in df.columns or y_column not in df.columns:
            print(f"Columns '{x_column}' or '{y_column}' not found in {csv_file}")
            continue

        # Plot the data
        if x_column == 'Timestep':
            plt.plot(df[x_column]*0.0005, df[y_column], label=label)  # convert timesteps to picoseconds (if necessary)
        else:
            plt.plot(df[x_column], df[y_column], label=label)

    # Add labels, legend, and grid
    if x_column == 'Timestep':
        plt.xlabel('Time (ps)')
        plt.title(f"{y_column} vs Time for various {label_column}")
    else:
        plt.xlabel(x_column)
        plt.title(f"{y_column} vs {x_column} for various {label_column}")
    plt.ylabel(y_column)
    plt.legend(title="Simulations")
    plt.grid(True)

    # Save or display the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


# Plots the specified columns (x_column, y_column) for a list of simulation IDs from a CSV file (usually all_simulations)
def plot_allsims_data(csv_file, sim_ids, x_column, y_column, output_file=None):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    sim_ids = [f"{sim_id:05}" for sim_id in sim_ids]


    # Check that required columns exist
    required_columns = [x_column, y_column, "Simulation ID"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_file}")

    # Filter the DataFrame for the specified simulation IDs
    df["Simulation ID"] = df["Simulation ID"].astype(str).str.zfill(5)  # Ensure IDs are 5 digits
    filtered_df = df[df["Simulation ID"].isin(sim_ids)]

    if filtered_df.empty:
        raise ValueError("No matching data found for the specified simulation IDs.")

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Loop through each simulation ID and plot its data
    x_data = []
    y_data = []
    for sim_id in sim_ids:
        sim_data = filtered_df[filtered_df["Simulation ID"] == sim_id]  # all of the data for that sim_id
        if sim_data.empty:
            print(f"No data found for Simulation ID {sim_id}")
            continue
        x_data.append(np.abs(sim_data[x_column].iloc[0]))  # get the x data you want
        y_data.append(np.abs(sim_data[y_column].iloc[0]))  # get the y data you want

    # Convert to numpy arrays for easier manipulation
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Get the sorted indices of x_data
    sorted_indices = np.argsort(x_data)

    # Rearrange x_data and y_data using the sorted indices
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices]

    plt.scatter(x_data, y_data)  # Scatter plot for each sim ID
    plt.plot(x_data, y_data, alpha=0.6)  # Trend line for each sim ID
    plt.xscale("log")
    # plt.axvline(x=1e-3, color='r', linestyle='--', linewidth=1, label='Park Suggested Strain Rate')
    # Add labels, title, and grid
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    # plt.legend()
    plt.title(f"{y_column} vs {x_column}")
    plt.grid(True)

    # Save or display the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


# Generates a list of file names following the naming convention when given the start and end id's (includes end id)
def generate_csv_list(start_id, end_id, folder="simulation_data/deform_data", prefix="sim", suffix=".csv"):
    file_list = []
    for sim_id in range(start_id, end_id + 1):
        # Format the simulation ID as 5 digits
        sim_id_str = f"{sim_id:05}"  # change the id to a 5 digit string
        
        file_name = f"{folder}/sim{sim_id_str}/{prefix}{sim_id_str}{suffix}"  # Construct the full file path
        file_list.append(file_name)
    return file_list


if __name__ == "__main__":
    main()