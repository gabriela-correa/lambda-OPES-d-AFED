#-----------------------------------------------------------------------------------------

import figstyle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openmm import unit

#-----------------------------------------------------------------------------------------
# INPUTS

T_psi_s = 1000.0  # K
T_phi_s = 1000.0  # K
T_lbda = 298.15   # K
T_atoms = 298.15  # K

# R in kJ/mol/K and kcal/mol/K
k_input = (unit.MOLAR_GAS_CONSTANT_R).value_in_unit(
    unit.kilojoules_per_mole / unit.kelvin
)
k_output = (unit.MOLAR_GAS_CONSTANT_R).value_in_unit(
    unit.kilocalories_per_mole / unit.kelvin
)

eq_size = 5000      # 1 ns
block_size = 5000   # 1 ns
block_time = 1    # ns

kappa_phi = 1000.0
kappa_psi = 1000.0

# Number of bins for lambda integration
n_lambda_bins = 30

T_ext_dihedral = T_phi_s

#-----------------------------------------------------------------------------------------
# AUXILIARY FUNCTIONS

def wrap_angle_rad(angle):
    """Map angle in radians to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a, b):
    """Periodic difference a - b for angles in radians."""
    return a - b - 2 * np.pi * np.round((a - b) / (2 * np.pi))


def S22(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else (6 * x**2 - 15 * x + 10) * x**3)


def S22p(x):
    return 0.0 if x < 0 else (0.0 if x > 1 else 30 * (x**2 - 2 * x + 1) * x**2)


def S32(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else 10 * x**6 - 24 * x**5 + 15 * x**4)


def S32p(x):
    return 0.0 if x < 0 else (0.0 if x > 1 else 6 * 10 * x**5 - 5 * 24 * x**4 + 4 * 15 * x**3)


#-----------------------------------------------------------------------------------------
# DATASET

files = ["../output.csv"]
dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data.drop(index=range(eq_size), inplace=True)
data.index = np.arange(0, len(data))

data["time"] = data['#"Step"'] * 1e-6
data["lbda"] = 1.0 - np.abs(1.0 - data["theta (dimensionless)"])

data["UsoftLJ"] = data["UsoftLJ (kJ/mol)"]
data["ULJCoul"] = data["ULJCoul (kJ/mol)"]

data["phi_s"] = data["phi_s (rad)"]
data["psi_s"] = data["psi_s (rad)"]
data["phi"] = data["phi (rad)"]
data["psi"] = data["psi (rad)"]

files = ["../opes_bias.csv"]
dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

data2 = pd.concat(dfs, ignore_index=True)
data2.drop(index=range(eq_size), inplace=True)
data2.index = np.arange(0, len(data2))

nblocks = len(data) // block_size

#-----------------------------------------------------------------------------------------
# dU/dlambda

start = dict(A=0.0, B=0.35)
finish = dict(A=1.0, B=1.0)

data["hA"] = 0.0
data["hB"] = 0.0
data["hpA"] = 0.0
data["hpB"] = 0.0

for state in range(len(data)):
    lam = data["lbda"][state]

    xA = (lam - start["A"]) / (finish["A"] - start["A"])
    xB = (lam - start["B"]) / (finish["B"] - start["B"])

    data.at[state, "hA"] = S22(xA)
    data.at[state, "hpA"] = S22p(xA) * (1 / (finish["A"] - start["A"]))
    data.at[state, "hB"] = S32(xB)
    data.at[state, "hpB"] = S32p(xB) * (1 / (finish["B"] - start["B"]))

# dU/dlambda in kJ/mol
data["Up"] = data["hpA"] * data["UsoftLJ"] + data["hpB"] * (data["ULJCoul"] - data["UsoftLJ"])

# Extended-variable forces, if needed for diagnostics
data["force_phi_s"] = kappa_phi * angle_diff(data["phi"], data["phi_s"])
data["force_psi_s"] = kappa_psi * angle_diff(data["psi"], data["psi_s"])

#-----------------------------------------------------------------------------------------
# TEMPERATURE

data["T_lbda"] = data["T[theta] (K)"].expanding().mean()
data["T_phi_s"] = data["T[phi_s] (K)"].expanding().mean()
data["T_psi_s"] = data["T[psi_s] (K)"].expanding().mean()
data["T_atoms"] = data["Temperature (K)"].expanding().mean()

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["T_lbda"], "k-")
ax.plot(np.array([0, data["time"].iloc[-1]]), np.array([T_lbda, T_lbda]), "k--")
ax.set_xlabel("time (ns)")
ax.set_ylabel(r"$T$ (K)")
ax.set_ylim(T_lbda - 100, T_lbda + 100)
fig.savefig("temperature_lambda.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["T_phi_s"], "k-")
ax.plot(np.array([0, data["time"].iloc[-1]]), np.array([T_phi_s, T_phi_s]), "k--")
ax.set_xlabel("time (ns)")
ax.set_ylabel(r"$T$ (K)")
ax.set_ylim(T_phi_s - 500, T_phi_s + 500)
fig.savefig("temperature_phi_s.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["T_psi_s"], "k-")
ax.plot(np.array([0, data["time"].iloc[-1]]), np.array([T_psi_s, T_psi_s]), "k--")
ax.set_xlabel("time (ns)")
ax.set_ylabel(r"$T$ (K)")
ax.set_ylim(T_psi_s - 500, T_psi_s + 500)
fig.savefig("temperature_T_psi_s.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["T_atoms"], "k-")
ax.plot(np.array([0, data["time"].iloc[-1]]), np.array([T_atoms, T_atoms]), "k--")
ax.set_xlabel("time (ns)")
ax.set_ylabel(r"$T$ (K)")
ax.set_ylim(T_atoms - 100, T_atoms + 100)
fig.savefig("temperature_atoms.png", dpi=300)

#-----------------------------------------------------------------------------------------
# BIAS VS LAMBDA

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), sharex=True, dpi=300)
ax.plot(data["lbda"], (k_output / k_input) * data2["opes_conv_bias (kJ/mol)"], "bo", label="")
ax.plot(data["lbda"], (k_output / k_input) * data2["opes_expl_bias (kJ/mol)"], "ro", label="")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$U_\text{bias}$ (kcal/mol)")
fig.savefig("bias.png", dpi=300)

#-----------------------------------------------------------------------------------------
# LAMBDA / PHI / PSI VS TIME

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["lbda"], "o", markersize=0.05)
ax.set_ylabel(r"$\lambda$")
ax.set_xlabel(r"time (ns)")
fig.savefig("lambda_time.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["phi_s"], "o", markersize=0.05)
ax.set_ylabel(r"$\phi_s$ (rad)")
ax.set_xlabel(r"time (ns)")
fig.savefig("phis_time.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["psi_s"], "o", markersize=0.05)
ax.set_ylabel(r"$\psi_s$ (rad)")
ax.set_xlabel(r"time (ns)")
fig.savefig("psis_time.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["phi"], "o", markersize=0.05)
ax.set_ylabel(r"$\phi$ (rad)")
ax.set_xlabel(r"time (ns)")
fig.savefig("phi_time.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data["time"], data["psi"], "o", markersize=0.05)
ax.set_ylabel(r"$\psi$ (rad)")
ax.set_xlabel(r"time (ns)")
fig.savefig("psi_time.png", dpi=300)


#-----------------------------------------------------------------------------------------
# FREE ENERGY

# Lambda bins for thermodynamic integration
bins = np.linspace(0, 1, n_lambda_bins + 1)
labels = 0.5 * (bins[1:] + bins[:-1])
labels_full = np.concatenate(([0.0], labels, [1.0]))
dx_full = labels_full[1:] - labels_full[:-1]

def integrate_mean_force(y):
    """
    Integrate <dU/dlambda> over lambda by trapezoidal rule.

    If any lambda bin is missing, return NaNs for this whole block.
    This is intentional: a missing lambda bin cannot be replaced by the old estimator
    without mixing two different estimators.
    """

    y = np.asarray(y, dtype=float)
    y_full = np.concatenate(([0.0], y, [0.0]))

    if np.any(~np.isfinite(y_full)):
        return np.full_like(labels_full, np.nan, dtype=float)

    deltaF_full = np.zeros_like(labels_full, dtype=float)
    deltaF_full[1:] = np.cumsum(0.5 * (y_full[1:] + y_full[:-1]) * dx_full)

    # Convert kJ/mol to kcal/mol
    return (k_output / k_input) * deltaF_full


deltaFij_original = np.full((nblocks, labels_full.size), np.nan)

mean_Up_original_last = None

for i in range(nblocks):
    subset = data.iloc[: (i + 1) * block_size].copy()

    subset["lambda_bin"] = pd.cut(
        subset["lbda"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    mean_Up_original = (
        subset.groupby("lambda_bin", observed=False)["Up"]
        .mean()
        .reindex(labels)
        .values
    )

    deltaFij_original[i, :] = integrate_mean_force(mean_Up_original)

    mean_Up_original_last = mean_Up_original.copy()

#-----------------------------------------------------------------------------------------
# PLOTS: FREE ENERGY PROFILE AND MEAN FORCE

fig, ax = plt.subplots(1, 1, figsize=(2.3, 2.0), dpi=300)
ax.plot(labels_full, deltaFij_original[-1, :], "k-", lw=1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
ax.legend(fontsize=7, frameon=False)
fig.savefig("free_energy_profile.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.3, 2.0), dpi=300)
ax.plot(
    labels_full,
    (k_output / k_input) * np.concatenate(([0.0], mean_Up_original_last, [0.0])),
    "k-",
    lw=1.0,
)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\langle \partial U / \partial \lambda \rangle$ (kcal/mol)")
ax.legend(fontsize=7, frameon=False)
fig.savefig("mean_force.png", dpi=300)

#-----------------------------------------------------------------------------------------
# CONVERGENCE

print(f"deltaG: {deltaFij_original[-1, -1]:.2f} kcal/mol")

time = np.arange(1, int(len(deltaFij_original[:, -1])) + 1) * block_time

fig, ax = plt.subplots(1, 1, figsize=(2.3, 2.0), dpi=300)
ax.plot(time, deltaFij_original[:, -1], "k-", lw=1.0)
ax.set_xlabel(r"time (ns)")
ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
fig.savefig("free_energy_time.png", dpi=300)

np.savez(
    "block_deltaF.npz",
    deltaF=deltaFij_original[:, -1],
    deltaF_profile=deltaFij_original,
    mean_Up_kJmol=mean_Up_original_last,
    lambda_centers=labels,
    lambda_full=labels_full,
    time_blocks=time,
)

