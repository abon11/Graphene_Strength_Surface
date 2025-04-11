import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # initialize the plots for this sheet's strength surface (if user wants)
    figx, ax = plt.subplots()
    ax.set_xlabel(r'$\sigma_1$')
    ax.set_ylabel(r'$\sigma_2$')
    ax.set_title(f'Molecular Strength Surface of SV-Defective Graphene (x-dominant)')

    figy, ay = plt.subplots()
    ay.set_xlabel(r'$\sigma_1$')
    ay.set_ylabel(r'$\sigma_2$')
    ay.set_title(f'Molecular Strength Surface of SV-Defective Graphene (y-dominant)')

    filepath = "/data1/avb25/graphene_sim_data"

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    s1 = np.array([
        18.365, 18.412, 15.397, 14.006, 12.429, 0, 0, 0, 0,
        -16.788, -23.467, -32.603, -43.362, -55.374, -58.481, -60.058,
        -60.707, -62.377, -67.246, -67.71, -80.325, -82.319, -76.475,
        -74.852, -72.719, 19.958, 25.527, 20.422, 26.92, 23.713,
        24.093, 25.401, 26.498, 27.595, 25.443, 26.456, 23.924,
        22.025, 17.806, 17.215, 16.076, 12.194, 14.388, 10.295,
        8.4388, 5.9916, 0, 0, 0, 0
    ])

    s2 = np.array([
        19.958, 25.527, 20.422, 26.92, 23.713, 24.093, 25.401, 26.498, 27.595,
        25.443, 26.456, 23.924, 22.025, 17.806, 17.215, 16.076, 12.194, 14.388,
        10.295, 8.4388, 5.9916, 0, 0, 0, 0, 18.365, 18.412, 15.397, 14.006,
        12.429, 0, 0, 0, 0, -16.788, -23.467, -32.603, -43.362, -55.374,
        -58.481, -60.058, -60.707, -62.377, -67.246, -67.71, -80.325,
        -82.319, -76.475, -74.852, -72.719
    ])

    ax.scatter(s1 / 1000, s2 / 1000, color='black', label='Sato Data')
    ay.scatter(s1 / 1000, s2 / 1000, color='black', label='Sato Data')

    plot_group(ax, list(range(203, 214)), colors[0], label='60x60 pristine')  # x
    plot_group(ay, list(range(280, 291)), colors[0], label='60x60 pristine')  # y

    # # plot the first two iterations outside for the legend
    # plot_group(ax, makelist(25), colors[9], label='60x60 0.5% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")
    # plot_group(ay, makelist(36), colors[9], label='60x60 0.5% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")

    # for i in range(47, 179, 22):
    #     plot_group(ax, makelist(i), colors[9], full_csv=f"{filepath}/defected_data/all_simulations.csv")
    #     plot_group(ay, makelist(i+11), colors[9], full_csv=f"{filepath}/defected_data/all_simulations.csv")


    # # 1%
    # plot_group(ax, makelist(179), colors[9], label='60x60 1% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")
    # plot_group(ay, makelist(190), colors[9], label='60x60 1% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")

    # for i in range(179, 333, 22):
    #     plot_group(ax, makelist(i), colors[9], full_csv=f"{filepath}/defected_data/all_simulations.csv")
    #     plot_group(ay, makelist(i+11), colors[9], full_csv=f"{filepath}/defected_data/all_simulations.csv")

    
    # 2%
    plot_group(ax, makelist(333), colors[9], label='60x60 2% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")
    plot_group(ay, makelist(344), colors[9], label='60x60 2% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")

    for i in range(333, 487, 22):
        plot_group(ax, makelist(i), colors[9], full_csv=f"{filepath}/defected_data/all_simulations.csv")
        plot_group(ay, makelist(i+11), colors[9], full_csv=f"{filepath}/defected_data/all_simulations.csv")





    plot_group(ax, list(range(70, 81)), colors[3], label='100x100 pristine')  # x
    plot_group(ay, list(range(192, 203)), colors[3], label='100x100 pristine')  # y

    # # plot the first two iterations outside for the legend
    # plot_group(ax, makelist(487), colors[6], label='100x100 0.5% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")
    # plot_group(ay, makelist(498), colors[6], label='100x100 0.5% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")

    # for i in range(487, 641, 22):
    #     plot_group(ax, makelist(i), colors[6], full_csv=f"{filepath}/defected_data/all_simulations.csv")
    #     plot_group(ay, makelist(i+11), colors[6], full_csv=f"{filepath}/defected_data/all_simulations.csv")


    # # 1%
    # plot_group(ax, makelist(641), colors[6], label='100x100 1% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")
    # plot_group(ay, makelist(652), colors[6], label='100x100 1% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")

    # for i in range(641, 795, 22):
    #     plot_group(ax, makelist(i), colors[6], full_csv=f"{filepath}/defected_data/all_simulations.csv")
    #     plot_group(ay, makelist(i+11), colors[6], full_csv=f"{filepath}/defected_data/all_simulations.csv")

    
    # 2%
    plot_group(ax, makelist(795), colors[6], label='100x100 2% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")
    plot_group(ay, makelist(806), colors[6], label='100x100 2% SV', full_csv=f"{filepath}/defected_data/all_simulations.csv")

    for i in range(795, 949, 22):
        plot_group(ax, makelist(i), colors[6], full_csv=f"{filepath}/defected_data/all_simulations.csv")
        plot_group(ay, makelist(i+11), colors[6], full_csv=f"{filepath}/defected_data/all_simulations.csv")


    ax.set_xlim(-15, 130)
    ax.set_ylim(-15, 130)
    ax.plot([-50, 130], [0, 0], color='black')
    ax.plot([0, 0], [-50, 130], color='black')
    ax.legend()

    ay.set_xlim(-15, 130)
    ay.set_ylim(-15, 130)
    ay.plot([-50, 130], [0, 0], color='black')
    ay.plot([0, 0], [-50, 130], color='black')
    ay.legend()


    figx.savefig(f"{filepath}/defected_data/Strength_Surface_sizetestx2SATO.png")
    figy.savefig(f"{filepath}/defected_data/Strength_Surface_sizetesty2SATO.png")


# makes a list of 10 numbers
def makelist(start):
    ans = list(range(start, start+11))
    return ans


def plot_group(ax, data_list, color, marker=None, label=None, full_csv='/data1/avb25/graphene_sim_data/pristine_data/all_simulations.csv'):
    str1, str2 = get_sim_data(full_csv, data_list)
    if marker is None:
        if label is not None:
            ax.scatter(str1[0], str2[0], color=color, label=label)

        for i in range(len(str1)):
            ax.scatter(str1[i], str2[i], color=color)
            ax.scatter(str2[i], str1[i], color=color)
    
    else:
        if label is not None:
            ax.scatter(str1[0], str2[0], color=color, marker=marker, label=label)

        for i in range(len(str1)):
            ax.scatter(str1[i], str2[i], color=color, marker=marker)
            ax.scatter(str2[i], str1[i], color=color, marker=marker)        


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