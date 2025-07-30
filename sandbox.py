import pandas as pd
import json
import local_config

# Load the CSV
df = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")

# Function to build the defect dictionary
def build_defect_dict(row):
    dtype = row.get("Defect Type")
    dperc = row.get("Defect Percentage")

    if pd.isna(dtype) or pd.isna(dperc):
        return json.dumps({})
    
    dtype = str(dtype).strip().upper()
    try:
        dperc = float(dperc)
    except ValueError:
        return json.dumps({})

    return json.dumps({dtype: dperc})

# Create new 'Defects' column
df["Defects"] = df.apply(build_defect_dict, axis=1)

# Drop old columns
df.drop(columns=["Defect Type", "Defect Percentage"], inplace=True)

# Reorder columns: insert "Defects" between "Theta" and "Defect Random Seed"
cols = df.columns.tolist()
try:
    theta_idx = cols.index("Theta")
    # Remove 'Defects' from the end
    cols.remove("Defects")
    # Insert 'Defects' right after 'Theta'
    cols.insert(theta_idx + 1, "Defects")
    df = df[cols]
except ValueError:
    print("⚠️ Column 'Theta' not found. 'Defects' will be at the end.")

# Save to a new CSV
df.to_csv("converted_output.csv", index=False)

print("✅ CSV conversion complete. Output saved to 'converted_output.csv'")
