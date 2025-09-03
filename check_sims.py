"""
This script checks the simulations in all_simulations.csv to ensure that everything we think is there is actually there.
For instance, if we ran 10k sims over a weekend, we can run this to see if any of the expected sims didn't save for any reason.
"""

from filter_csv import filter_data
import pandas as pd
import local_config

def main():
    df = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")
    run_check(df, "{\"SV\": 0.5}", (0, 100, 1), (0, 90, 10))
    run_check(df, "{\"SV\": 0.25, \"DV\": 0.25}", (0, 100, 1), (0, 90, 10))
    run_check(df, "{\"DV\": 0.5}", (0, 100, 1), (0, 90, 10))


def run_check(df, defects, seeds, thetas):
    """
    - The user puts inclusive ranges (because thats how it works for the filter already), however iterating
    in range of the tuple is exclusive, so the first thing we do here is generate a new tuple that has +1
    of the old tuple, making it inclusive for what the user put even when iterating
    """
    problems = 0
    seeds_incl = (seeds[0], seeds[1] + 1, seeds[2])
    thetas_incl = (thetas[0], thetas[1] + 1, thetas[2])

    # iterate through every seed (note that the * unpacks the tuple)
    for seed in range(*seeds_incl):
        exact_filters = {"Defects": defects, "Defect Random Seed": seed}
        range_filters = {"Theta Requested": (thetas[0], thetas[1])}

        # multiply by 10 because we expect there to be 10 "ratios" for each theta and seed (without biaxial)
        expected_length = max(0, (thetas[1] - thetas[0]) // thetas[2] + 1) * 10
        
        filtered_df = filter_data(df, exact_filters=exact_filters, range_filters=range_filters, remove_biaxial=True, remove_dupes=True)

        if len(filtered_df) != expected_length:
            for theta in range(*thetas_incl):
                exact_filters = {"Defects": defects, "Defect Random Seed": seed, "Theta Requested": theta}
                filtered_df = filter_data(df, exact_filters=exact_filters, remove_biaxial=True, remove_dupes=True)
                if len(filtered_df) != 10:
                    problems += 1
                    print(f"{defects}, seed {seed}, theta {theta} has length of {len(filtered_df)}")

        # Now specifically check the biaxial case (no theta restriction)
        exact_filters = {"Defects": defects, "Defect Random Seed": seed, "Strain Rate x": 0.001, "Strain Rate y": 0.001, "Strain Rate xy": 0}
        filtered_df = filter_data(df, exact_filters=exact_filters, remove_dupes=True)
        if len(filtered_df) != 1:
            problems += 1
            print(f"{defects}, seed {seed} has no biaxial data.")

    if problems == 1:
        print(f"Ended the check of {defects}, {seeds}, {thetas} with 1 problem encountered.")
    else:
        print(f"Ended the check of {defects}, {seeds}, {thetas} with {problems} problems encountered.")

if __name__ == "__main__":
    main()