
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

specimens = {
    2: {"height": 19.14, "area": 388.96},
    3: {"height": 18.80, "area": 378.43},
    4: {"height": 19.17, "area": 367.10},
    5: {"height": 19.10, "area": 392.03},
    6: {"height": 19.89, "area": 335.75},
}

plt.figure(figsize=(9, 6))

for trial, dimensions in specimens.items():
    df = pd.read_csv(
        DATA_DIR / f"Dog compression_{trial}.csv",
        skiprows=[1]
    )

    displacement = df["Displacement"]
    force = df["Force"] * 1000

    strain = (
        displacement - displacement.iloc[0]
    ) / dimensions["height"]

    stress = force / dimensions["area"]

    plt.plot(strain, stress, label=f"Trial {trial}")

plt.xlim(0, 0.05)
plt.xlabel("Engineering Compressive Strain (mm/mm)")
plt.ylabel("Engineering Compressive Stress (MPa)")
plt.title("Initial Stress–Strain Behavior of Sawbones Foam")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = FIGURES_DIR / "elastic_region_inspection.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved: {output_path}")
