
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Had to replace instron output dimensions with actual dimensions
specimens = {
    2: {"height": 19.14, "area": 388.96},
    3: {"height": 18.80, "area": 378.43},
    4: {"height": 19.17, "area": 367.10},
    5: {"height": 19.10, "area": 392.03},
    6: {"height": 19.89, "area": 335.75},
}

plt.figure(figsize=(9, 6))

for trial, dimensions in specimens.items():
    file_path = DATA_DIR / f"Dog compression_{trial}.csv"

    df = pd.read_csv(file_path, skiprows=[1])
    df.columns = df.columns.str.strip()

    displacement = pd.to_numeric(
        df["Displacement"], errors="coerce"
    )
    force = pd.to_numeric(
        df["Force"], errors="coerce"
    ) * 1000  # kN to N

    valid = displacement.notna() & force.notna()
    displacement = displacement[valid].reset_index(drop=True)
    force = force[valid].reset_index(drop=True)

    delta_L = displacement - displacement.iloc[0]

    # Strain (mm/mm)
    strain = delta_L / dimensions["height"]

    # Stress (N/mm² = MPa)
    stress = force / dimensions["area"]

    plt.plot(
        strain,
        stress,
        linewidth=1.8,
        label=f"Trial {trial}"
    )

plt.xlabel("Engineering Compressive Strain (mm/mm)")
plt.ylabel("Engineering Compressive Stress (MPa)")
plt.title("Stress–Strain Behavior of 40 PCF Sawbones Foam")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = FIGURES_DIR / "stress_strain.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Figure saved to: {output_path}")
