"""
This filters the all_simulations csv based on whatever fields you want (very helpful)
The big function here, filter_data, is used in a bunch of other scripts
"""

import pandas as pd
import local_config
import json
import numpy as np
import ast


def main():
    # ========== USER INTERFACE ==========
    folder = f'{local_config.DATA_DIR}/rotation_tests'
    csv_file = f"{folder}/all_simulations.csv"

    exact_filters = {
        "Num Atoms x": 60,
        "Num Atoms y": 60,
        # "Defects": '{"DV": 0.5}',  # will match NaN or "None"
        "Defects": "None",
        # "Defect Random Seed": 77,
        # "Theta Requested": 90,
        # "Strain Rate x": 0.001,
        # "Strain Rate y": 0.001,
        # "Strain Rate xy": 0.0
    }

    range_filters = {
        # "Defect Random Seed": (0, 19)
        # "Theta Requested": (90, 90),
        # "Sigma_1": (4, 20)
        # "Theta": (24, 32)
    }

    or_filters = {
        # "Defects": ["{\"DV\": 0.25, \"SV\": 0.25}", "{\"DV\": 0.5}", "{\"SV\": 0.5}"],
        # "Strain Rate x": [-0.00005, -0.00006]
        "Theta Requested": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    }

    uniaxial = False

    # ====================================
    
    df = pd.read_csv(csv_file)
    filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, or_filters=or_filters, remove_nones=True,
                              only_uniaxial=uniaxial, remove_biaxial=False, remove_dupes=True, duplic_freq=(0, 91, 100))
    
    print(f"Filtered {len(filtered_df)} rows from {len(df)} total.")
    filtered_df.to_csv("filtered.csv", index=False)
    print("Saved filtered data to filtered.csv")



# filters the dataset to whatever exact, range, or or filters we want
# when flip_strengths is true, it returns a dataset where every datapoint is duplicated but sig_1 is flipped with sig_2 
# duplic_freq is a tuple meaning (start_theta, end_theta, how many to jump by) to duplicate biaxial tension across all thetas
def filter_data(df, exact_filters=None, range_filters=None, or_filters=None, 
                flip_strengths=False, duplic_freq=None, only_uniaxial=False,
                remove_biaxial=False, remove_dupes=False, shift_theta=True, remove_nones=False):

    """
    Filter df on exact, range, OR filters and optionally flip strength directions.
    """
    filtered = df.copy()

    if remove_biaxial:
        filtered = drop_biaxial_rows(filtered)

    if duplic_freq is not None:
        filtered = duplicate_biaxial_rows(filtered, duplic_freq)

    filtered = alphabetize_dict(filtered, "Defects")

    # Apply exact filters
    for col, val in (exact_filters or {}).items():
        if col == "Defects":
            if val == "None" or val is None:
                val_dict = {}
            else:
                val_dict = parse_defects_json(val)

            def defect_match(row_json):
                return parse_defects_json(row_json) == val_dict

            filtered = filtered[filtered["Defects"].apply(defect_match)]

        else:
            filtered = filtered[filtered[col] == val]

    # Apply range filters
    for col, (mn, mx) in (range_filters or {}).items():
        filtered = filtered[(filtered[col] >= mn) & (filtered[col] <= mx)]

    # Apply OR filters
    for col, vals in (or_filters or {}).items():
        if col == "Defects":
            parsed_vals = []
            include_empty = False
            for v in vals:
                if v == "None" or v is None:
                    include_empty = True
                else:
                    parsed_vals.append(parse_defects_json(v))

            def or_defect_match(row_json):
                parsed_row = parse_defects_json(row_json)
                return parsed_row in parsed_vals or (include_empty and parsed_row == {})

            filtered = filtered[filtered["Defects"].apply(or_defect_match)]

        else:
            filtered = filtered[filtered[col].isin(vals)]

    if only_uniaxial:
        filtered = get_uniaxial_tension(filtered)

    # Optionally double the strength data
    if flip_strengths:
        filtered = flip_strength_vals(filtered)

    if remove_dupes:
        filtered = drop_duplicates(filtered)

    if shift_theta is True:
        filtered["Theta"] = filtered["Theta"] - filtered["Rotation Angle"]
        print("Shifted Theta such that Theta[new] = Theta[old] - Rotation Angle.")

    if remove_nones:
        filtered = drop_nones(filtered)
    
    return filtered


def alphabetize_dict(df: pd.DataFrame, column: str, inplace: bool = False) -> pd.DataFrame:
    """
    Alphabetize keys in a JSON-formatted string column for every row.
    Leaves NaN as is, accepts JSON strings or dicts, and returns a DataFrame.
    """
    def to_canonical(obj):
        # Leave NaN or None unchanged
        if obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return obj

        d = None
        if isinstance(obj, dict):
            d = obj
        elif isinstance(obj, str):
            s = obj.strip()
            # Try JSON
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                # Try Python dict literal
                try:
                    d = ast.literal_eval(s)
                except Exception:
                    # Could not parse, return original
                    return obj
        else:
            # Unknown type, return original
            return obj

        if not isinstance(d, dict):
            return obj

        # Dump with alphabetical keys and consistent spacing like {"DV": 0.25, "SV": 0.25}
        return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(',', ': '))

    target = df if inplace else df.copy()
    target[column] = target[column].apply(to_canonical)
    return target


# Delete all rows that have None (such as one that never broke or terminated early for some reason). Good for check_sims.py
def drop_nones(df):
    cols = ["Strength_1", "Strength_2", "Strength_3"]
    mask = df[cols].isna().any(axis=1)
    return df.loc[~mask].copy()    

# Delete all biaxial rows from the df (important for the check_sims.py file)
def drop_biaxial_rows(df):
    mask = (df["Strain Rate x"] == df["Strain Rate y"]) & (df["Strain Rate xy"] == 0)
    return df.loc[~mask].copy()

def get_uniaxial_tension(df, threshold=0.2):
    """
    For each (seed, theta, defects), pick the row with the smallest Strength_2 / Strength_1
    subject to the ratio being below `threshold`. Returns a new DataFrame.
    """
    # Compute ratio safely and filter
    ratio = df["Strength_2"] / df["Strength_1"]
    mask = (ratio < threshold) & np.isfinite(ratio)
    in_thresh = df.loc[mask].copy()
    in_thresh["ratio"] = ratio.loc[in_thresh.index]

    # Get index of minimal ratio per (seed, theta)
    idx_min = (in_thresh.groupby(["Defect Random Seed", "Theta Requested", "Defects"], as_index=False)["ratio"].idxmin())

    # Collect rows and sort
    result = (in_thresh.loc[idx_min["ratio"]].sort_values(["Defect Random Seed", "Theta Requested", "Defects"]).reset_index(drop=True))

    # Do not want the ratio column in the output
    result = result.drop(columns=["ratio"])

    return result

# helper for filter, parses the defects json
def parse_defects_json(json_str):
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, json.JSONDecodeError):
        pass
    return {}

# helper for filter, flips and copies the strength values
def flip_strength_vals(df):
    flipped = df.copy()
    flipped["Strength_1"], flipped["Strength_2"] = df["Strength_2"], df["Strength_1"]
    return pd.concat([df, flipped], ignore_index=True)

# duplicates the pure biaxial tensile rows into all theta categories we care about
# duplication_frequency is a tuple that represents (start_theta, end_theta, how many to jump by)
# ex) duplication_frequency = (0, 31, 5) means we would duplicate the biaxial tensile tests to thetas of 0, 5, 10, 15, 20, 25, and 30.
def duplicate_biaxial_rows(df, duplication_frequency):
    # Identify perfectly biaxial tension cases (when the strain rates equal each other when shear is zero)
    is_biaxial = ((df["Theta Requested"] == -1) | ((df["Strain Rate x"] == df["Strain Rate y"]) & (df["Strain Rate xy"] == 0)))

    # Extract the biaxial rows
    biaxial_rows = df[is_biaxial]

    # For each theta, duplicate the biaxial row and assign that theta
    new_rows = []
    for theta in range(*duplication_frequency):
        if theta != -1:
            for _, row in biaxial_rows.iterrows():
                new_row = row.copy()
                new_row["Theta"] = theta
                new_row["Theta Requested"] = theta
                new_rows.append(new_row)

    df = df[df["Theta Requested"] != -1]  # remove the -1 so we don't plot it
    # Append duplicated rows to dataframe
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

def drop_duplicates(df):
    """
    This drops rows that are duplicates across all columns except 'Simulation ID',
    keeping the row with the smallest simid in each group.
    """
    subset = [c for c in df.columns if c not in {"Simulation ID", "Simulation Time"}]
    tmp = df.sort_values("Simulation ID", kind="mergesort")
    return tmp.drop_duplicates(subset=subset, keep="first").copy()

if __name__ == "__main__":
    main()
