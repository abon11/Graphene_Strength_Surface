#!/bin/bash
defs="mx"

# Fourier orders
for fo in $(seq 3 3); do

    # Lambda values
    for lam in 10; do
        echo "========================================"
        echo "Running Fourier order $fo with lam $lam"
        echo "========================================"

        # Run MPI job (BLOCKING)
        mpiexec -n 51 python3 fit_3D_DP.py --fourier_order "$fo" --lam "$lam" --defs "$defs"

        echo "MPI run finished for fo=$fo, lam=$lam"
        echo "Running cleanup Python script..."

        python3 - <<EOF
import glob, os
import pandas as pd

fourier_order = ${fo}
lam = ${lam}
lam_str = f"{lam:.0e}"
defs = "${defs}"

# Clean it: '1.0e+00' → '1e0'
lam_str = lam_str.replace(".0", "").replace("+0", "").replace("+", "").replace("-0", "-")

print(f"Saving z_{defs}{fourier_order}_reg{lam_str}.csv")

files = glob.glob(f"z_{defs}{fourier_order}_reg{lam_str}.csv_rank*.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(by="Defect Random Seed").reset_index(drop=True)

df.to_csv(f"z_{defs}{fourier_order}_reg{lam_str}.csv", index=False)

for f in files:
    os.remove(f)
EOF

        echo "Cleanup completed for fo=$fo, lam=$lam"
        echo ""
    done
done
