import pandas as pd
from plot_StrengthSurface import filter_data
import local_config


def main():

    # ========== USER INTERFACE ==========
    folder = f'{local_config.DATA_DIR}/defected_data'
    csv_file = f"{folder}/all_simulations.csv"

    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        "Defect Type": "None",  # will match NaN or "None"
        "Defect Percentage": 0,
        # "Defect Random Seed": 1,
        # "Theta": 90
    }

    range_filters = {
        # "Defect Percentage": (0.4, 0.6),
        # "Defect Random Seed": (1, 42),
        # "Theta": (0, 90),
    }

    or_filters = {
        # "Defect Type": ["SV", "DV"],
        "Theta": [0, 30, 60, 90]
    }
    # ====================================
    df = pd.read_csv(csv_file)
    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, dupe_thetas=False)
    
    print(f"Filtered {len(filtered_df)} rows from {len(df)} total.")
    filtered_df.to_csv("filtered.csv", index=False)
    print("Saved filtered data to filtered.csv")


# def filter_data(df, exact_filters=None, range_filters=None, or_filters=None):
#     filtered = df.copy()

#     # Apply exact filters (with special logic for "Defect Type" and "Theta")
#     if exact_filters:
#         for col, val in exact_filters.items():
#             if col == "Defect Type" and (val == "None" or val is None):
#                 mask = (filtered[col] == "None") | (filtered[col].isna())
#                 filtered = filtered[mask]
#             elif col == "Theta":
#                 isotropic_mask = filtered["Strain Rate x"] == filtered["Strain Rate y"]
#                 theta_mask = (filtered["Theta"] == val) & (~isotropic_mask)
#                 filtered = filtered[isotropic_mask | theta_mask]
#             else:
#                 filtered = filtered[filtered[col] == val]

#     # Apply range filters
#     if range_filters:
#         for col, (mn, mx) in range_filters.items():
#             filtered = filtered[(filtered[col] >= mn) & (filtered[col] <= mx)]

#     # Apply OR filters (special logic for "Defect Type")
#     if or_filters:
#         for col, vals in or_filters.items():
#             if col == "Defect Type":
#                 has_none = "None" in vals or None in vals
#                 val_set = set(vals) - {"None", None}
#                 mask = filtered[col].isin(val_set)
#                 if has_none:
#                     mask |= filtered[col].isna()
#                 filtered = filtered[mask]
#             else:
#                 filtered = filtered[filtered[col].isin(vals)]

#     return filtered


if __name__ == "__main__":
    main()
