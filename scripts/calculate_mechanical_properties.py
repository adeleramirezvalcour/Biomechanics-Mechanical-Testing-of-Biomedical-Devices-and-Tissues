
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

dimensions = {
    2: (19.14, 20.29, 19.17),
    3: (18.80, 19.19, 19.72),
    4: (19.17, 19.19, 19.13),
    5: (19.10, 19.17, 20.45),
    6: (19.89, 19.12, 17.56),
}

STRAIN_MIN = 0.01
STRAIN_MAX = 0.03
OFFSET = 0.002

results = []

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for plot_index, (trial, dims) in enumerate(dimensions.items()):
    height, width, thickness = dims
    area = width * thickness

    file_path = DATA_DIR / f"Dog compression_{trial}.csv"

    df = pd.read_csv(file_path, skiprows=[1])
    df.columns = df.columns.str.strip()

    displacement = pd.to_numeric(
        df["Displacement"], errors="coerce"
    ).to_numpy()

    force = pd.to_numeric(
        df["Force"], errors="coerce"
    ).to_numpy() * 1000

    valid = np.isfinite(displacement) & np.isfinite(force)
    displacement = displacement[valid]
    force = force[valid]

    displacement = displacement - displacement[0]
    strain = displacement / height
    stress = force / area

    fit_mask = (
        (strain >= STRAIN_MIN)
        & (strain <= STRAIN_MAX)
    )

    x = strain[fit_mask]
    y = stress[fit_mask]

    if len(x) < 3:
        raise ValueError(
            f"Trial {trial}: insufficient fitting data."
        )

    modulus, intercept = np.polyfit(x, y, 1)
    predicted = modulus * x + intercept

    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    stiffness = modulus * area / height

    offset_stress = modulus * (strain - OFFSET) + intercept
    difference = stress - offset_stress

    candidates = np.where(
        (strain[:-1] >= STRAIN_MAX)
        & (strain[1:] > strain[:-1])
        & (difference[:-1] >= 0)
        & (difference[1:] <= 0)
    )[0]

    yield_strain = np.nan
    yield_strength = np.nan
    energy_absorption = np.nan

    if len(candidates) > 0:
        i = candidates[0]
        denominator = difference[i] - difference[i + 1]

        if denominator != 0:
            fraction = difference[i] / denominator

            yield_strain = (
                strain[i]
                + fraction * (strain[i + 1] - strain[i])
            )

            yield_strength = (
                stress[i]
                + fraction * (stress[i + 1] - stress[i])
            )

            loading_strain = strain[:i + 1]
            loading_stress = stress[:i + 1]

            valid_loading = (
                np.isfinite(loading_strain)
                & np.isfinite(loading_stress)
                & (loading_strain >= 0)
            )

            loading_strain = loading_strain[valid_loading]
            loading_stress = loading_stress[valid_loading]

            increasing = np.concatenate((
                [True],
                np.diff(loading_strain) > 0
            ))

            loading_strain = loading_strain[increasing]
            loading_stress = loading_stress[increasing]

            if len(loading_strain) >= 2:
                energy_strain = np.append(
                    loading_strain, yield_strain
                )

                energy_stress = np.append(
                    loading_stress, yield_strength
                )

                energy_absorption = np.trapezoid(
                    energy_stress, energy_strain
                )

    results.append({
        "Trial": trial,
        "Stiffness (N/mm)": stiffness,
        "Modulus (MPa)": modulus,
        "Yield Strain": yield_strain,
        "Yield Strength (MPa)": yield_strength,
        "Energy Absorption (MJ/m³)": energy_absorption,
        "R²": r_squared,
    })

    ax = axes[plot_index]

    plot_limit = (
        max(0.08, yield_strain * 1.5)
        if np.isfinite(yield_strain)
        else 0.15
    )

    plot_mask = (
        (strain >= 0)
        & (strain <= plot_limit)
    )

    ax.plot(
        strain[plot_mask],
        stress[plot_mask],
        label="Experimental curve",
        linewidth=2,
    )

    ax.plot(
        strain[plot_mask],
        offset_stress[plot_mask],
        "--",
        label="0.2% offset line",
    )

    ax.plot(
        x,
        predicted,
        color="green",
        linewidth=3,
        label="Linear fit",
    )

    if np.isfinite(yield_strain):
        ax.scatter(
            yield_strain,
            yield_strength,
            color="red",
            s=60,
            zorder=5,
            label="Offset intersection",
        )

    ax.set_title(f"Trial {trial}")
    ax.set_xlabel("Engineering Strain (mm/mm)")
    ax.set_ylabel("Compressive Stress (MPa)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

axes[-1].axis("off")

fig.suptitle(
    "Preliminary 0.2% Offset Yield Analysis",
    fontsize=16,
)

plt.tight_layout()

figure_path = FIGURES_DIR / "offset_yield_analysis.png"

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

results_df = pd.DataFrame(results)

print("\nMECHANICAL PROPERTIES")
print(results_df.round(4).to_string(index=False))

numeric_columns = results_df.drop(columns="Trial")

print("\nMEAN")
print(numeric_columns.mean().round(4))

print("\nSTANDARD DEVIATION")
print(numeric_columns.std(ddof=1).round(4))

results_path = RESULTS_DIR / "mechanical_properties.csv"
results_df.to_csv(results_path, index=False)

print(f"\nResults saved to: {results_path}")
print(f"Figure saved to: {figure_path}")

if results_df["Yield Strength (MPa)"].isna().any():
    print("\nSome trials have no valid offset intersection.")

print("\nYield values are preliminary pending method validation.")
