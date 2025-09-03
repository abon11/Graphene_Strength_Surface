import numpy as np
import pandas as pd
from scipy.linalg import polar
import local_config
import warnings

ts = 5e-4  # timestep (set TS=1.0 if 'Fracture Time' is already in seconds)

df = pd.read_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations.csv")

def rotation_from_row(row, tol=1e-10):
    """
    Compute lattice rotation (deg) at fracture from engineering rates x time.
    Warn if R^T R != I or det(R) != 1 beyond tolerance.
    """
    # total engineering strains at fracture
    ex = row['Strain Rate x']  * row['Fracture Time'] * ts
    ey = row['Strain Rate y']  * row['Fracture Time'] * ts
    ez = row['Strain Rate z']  * row['Fracture Time'] * ts
    gxy = row['Strain Rate xy'] * row['Fracture Time'] * ts
    gxz = row['Strain Rate xz'] * row['Fracture Time'] * ts
    gyz = row['Strain Rate yz'] * row['Fracture Time'] * ts

    # Deformation gradient (engineering form)
    F = np.array([[1.0 + ex, gxy, gxz], [0.0, 1.0 + ey, gyz], [0.0, 0.0, 1 + ez]], dtype=float)

    # Right polar: F = R @ U
    R, U = polar(F, side='right')

    # Checks (orthogonality & determinant)
    orth_err = np.linalg.norm(R.T @ R - np.eye(3))
    detR = np.linalg.det(R)

    if orth_err > tol or abs(detR - 1.0) > tol:
        warnings.warn(
            f"Row {row.name}: rotation check failed "
            f"(||R^T R - I||_F={orth_err:.2e}, det(R)={detR:.12f}).",
            RuntimeWarning
        )

    # Lattice rotation angle (deg) from first column of R
    phi_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # this is purely in the xy plane.

    # Optionally compute effective angle if your CSV has it
    out = {'phi_deg': phi_deg}

    return pd.Series(out)

# Compute the angle per-row and name the series
rotation_series = df.apply(rotation_from_row, axis=1)
rotation_series.name = 'Rotation Angle'

# Attach and place the column between 'Theta' and 'Defects'
df['Rotation Angle'] = rotation_series
cols = list(df.columns)
if 'Theta' in cols and 'Defects' in cols:
    # remove then reinsert between Theta and Defects
    cols.remove('Rotation Angle')
    i_theta = cols.index('Theta')
    i_def   = cols.index('Defects')
    insert_pos = i_theta + 1 if i_theta < i_def else i_def
    cols.insert(insert_pos, 'Rotation Angle')
    df = df.reindex(columns=cols)

# Save if useful
df.to_csv(f"{local_config.DATA_DIR}/rotation_tests/all_simulations_2.csv", index=False)

print(df.head())

