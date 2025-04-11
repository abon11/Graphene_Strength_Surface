import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    folder = '/data1/avb25/graphene_sim_data/defected_data'
    # Specify columns to plot
    x_column = "Strain_1"
    y_column = "PrincipalStress_1"
    label_col = "Strain Rate x"

    # csv_files = generate_csv_list(1, 9)  # generate list of csvs to plot data you want
    # csv_files.insert(0, "simulation_data/sim00001_fulldata.csv")

    all_sims = "simulation_data/deform_data/all_simulations.csv"
    # Call the plotting function
    plot_detailed_data([f'{folder}/sim00400/dump.csv'], x_column, y_column, all_sims, label_col, output_file=f"{folder}combined_StressStrain.png")

    # Call the function to plot
    # plot_allsims_data(all_sims, list(range(1, 10)), 'Strain Rate x', 'Strength_1', output_file=f"{folder}strength_vs_StrainRate.png")


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