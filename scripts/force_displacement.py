
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.figure(figsize=(9, 6))

# Note: Only trials 2-6 were valid
for trial in range(2, 7):
    file_path = DATA_DIR / f"Dog compression_{trial}.csv"

    df = pd.read_csv(file_path, skiprows=[1])
    df.columns = df.columns.str.strip()

    displacement = pd.to_numeric(df["Displacement"], errors="coerce")
    force = pd.to_numeric(df["Force"], errors="coerce") * 1000

    valid = displacement.notna() & force.notna()

    plt.plot(
        displacement[valid],
        force[valid],
        label=f"Trial {trial}",
        linewidth=1.8
    )

plt.xlabel("Displacement (mm)")
plt.ylabel("Compressive Force (N)")
plt.title("Force–Displacement Behavior of 40 PCF Sawbones Foam")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = FIGURES_DIR / "force_displacement.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Figure saved to: {output_path}")
