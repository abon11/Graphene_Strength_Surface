import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # initialize the plots for this sheet's strength surface (if user wants)
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\sigma_1$')
    ax.set_ylabel(r'$\sigma_2$')
    ax.set_title(f'Molecular Strength Surface of Defected Graphene (x-dominant)')

    filepath = "/data1/avb25/graphene_sim_data"

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # THIS IS FOR X-DOMINANT
    # plot_group(ax, list(range(103, 114)), '40x40', colors[0])
    # plot_group(ax, list(range(115, 126)), '42x42', colors[1])
    # plot_group(ax, list(range(59, 70)), '50x50', colors[2])
    # plot_group(ax, list(range(126, 137)), '60x60', colors[3])
    # plot_group(ax, list(range(137, 148)), '80x80', colors[4])
    # plot_group(ax, list(range(70, 81)), '100x100', colors[5])

    # THIS IS FOR Y-DOMINANT
    # plot_group(ax, list(range(148, 159)), '40x40', colors[0])
    # plot_group(ax, list(range(159, 170)), '50x50', colors[2])
    # plot_group(ax, list(range(170, 181)), '60x60', colors[3])
    # plot_group(ax, list(range(181, 192)), '80x80', colors[4])
    # plot_group(ax, list(range(192, 203)), '100x100', colors[5])  # still running

    # plot_group(ax, list(range(103, 114)), '40x40, x-dominant', colors[0])
    # plot_group(ax, list(range(148, 159)), '40x40, y-dominant', colors[9])
    # plot_group(ax, list(range(126, 137)), '60x60, x-dominant', colors[3])
    # plot_group(ax, list(range(170, 181)), '60x60, y-dominant', colors[1])

    # str1, str2 = get_sim_data('simulation_data/deform_data/all_simulations.csv', list(range(159, 170)))
    # str1 = np.delete(np.append(str1, str1[0]), 0)
    # str2 = np.delete(np.append(str2, str2[0]), 0)

    # ax.plot(str1, str2, color='blue', label='Expected y-dominant')
    # ax.plot(str2, str1, color='blue')

    # str1, str2 = get_sim_data('simulation_data/deform_data/all_simulations.csv', list(range(59, 70)))
    # str1 = np.delete(np.append(str1, str1[0]), 0)
    # str2 = np.delete(np.append(str2, str2[0]), 0)

    # ax.plot(str1, str2, color='green', label='Expected x-dominant')
    # ax.plot(str2, str1, color='green')
    plot_group(ax, list(range(203, 214)), '60x60 pristine', colors[0])

    plot_group(ax, list(range(1, 12)), '60x60 hole', colors[1], full_csv=f"{filepath}/defected_data/all_simulations.csv")

    ax.set_xlim(-15, 130)
    ax.set_ylim(-15, 130)
    ax.plot([-50, 130], [0, 0], color='black')
    ax.plot([0, 0], [-50, 130], color='black')
    ax.legend()
    fig.savefig(f"{filepath}/defected_data/Strength_Surface_defx.png")


def plot_group(ax, data_list, label, color, full_csv='/data1/avb25/graphene_sim_data/pristine_data/all_simulations.csv'):
    str1, str2 = get_sim_data(full_csv, data_list)
    ax.scatter(str1[0], str2[0], color=color, label=label)

    for i in range(len(str1)):
        ax.scatter(str1[i], str2[i], color=color)
        ax.scatter(str2[i], str1[i], color=color)


def get_sim_data(csv_file, sim_ids, x_column="Strength_1", y_column="Strength_2"):
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

    return x_data, y_data
    

if __name__ == "__main__":
    main()