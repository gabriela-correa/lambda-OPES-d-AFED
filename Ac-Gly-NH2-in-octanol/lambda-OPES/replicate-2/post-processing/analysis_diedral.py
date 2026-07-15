#-----------------------------------------------------------------------------------------

import figstyle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

files = "../lbda_dihedral_NCCN.csv"
data = pd.read_csv(files)
data.drop(index=range(0), inplace=True)
lbda = data["lbda"].values 
phi_deg = data["phi_deg"].values

fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharey=True)
axes.plot(np.arange(len(phi_deg))*0.0002, phi_deg, 'o', markersize=0.05)
axes.set_ylabel("Dihedral (deg)")
axes.set_xlabel("time (ns)")
plt.savefig("dihedral_time_NCCN.png", dpi=300)

phi_min, phi_max = -180, 180
lam_bins = np.linspace(0.0, 1.0, 21)
phi_bins = np.linspace(phi_min, phi_max, 91)

# bins escolhidos
selected_bins = [0, 10, 19]

fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)

for ax, i in zip(axes, selected_bins):
    lo, hi = lam_bins[i], lam_bins[i+1]
    mask = (lbda >= lo) & (lbda < hi if i < len(lam_bins)-2 else lbda <= hi)
    vals = phi_deg[mask]
    ax.hist(vals, bins=phi_bins, density=True, histtype="step")
    ax.set_xlim(phi_min, phi_max)

axes[0].set_ylabel("Probability density")
axes[0].set_xlabel("Dihedral (deg)")
axes[1].set_xlabel("Dihedral (deg)")
axes[2].set_xlabel("Dihedral (deg)")

axes[0].set_title(r"$\lambda=0.0$", fontsize=10)
axes[1].set_title(r"$\lambda=0.5$", fontsize=10)
axes[2].set_title(r"$\lambda=1.0$", fontsize=10)

plt.savefig("dihedral_NCCN.png", dpi=300)
plt.show()



files = "../lbda_dihedral_CNCC.csv"
data = pd.read_csv(files)
data.drop(index=range(0), inplace=True)
lbda = data["lbda"].values 
phi_deg = data["phi_deg"].values

fig, axes = plt.subplots(1, 1, figsize=(2, 2), sharey=True)
axes.plot(np.arange(len(phi_deg))*0.0002, phi_deg, 'o', markersize=0.05)
axes.set_ylabel("Dihedral (deg)")
axes.set_xlabel("time (ns)")
plt.savefig("dihedral_time_CNCC.png", dpi=300)


phi_min, phi_max = -180, 180
lam_bins = np.linspace(0.0, 1.0, 21)
phi_bins = np.linspace(phi_min, phi_max, 91)

# bins escolhidos
selected_bins = [0, 10, 19]

fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True)

for ax, i in zip(axes, selected_bins):
    lo, hi = lam_bins[i], lam_bins[i+1]
    mask = (lbda >= lo) & (lbda < hi if i < len(lam_bins)-2 else lbda <= hi)
    vals = phi_deg[mask]
    ax.hist(vals, bins=phi_bins, density=True, histtype="step")
    ax.set_xlim(phi_min, phi_max)

axes[0].set_ylabel("Probability density")
axes[0].set_xlabel("Dihedral (deg)")
axes[1].set_xlabel("Dihedral (deg)")
axes[2].set_xlabel("Dihedral (deg)")

axes[0].set_title(r"$\lambda=0.0$", fontsize=10)
axes[1].set_title(r"$\lambda=0.5$", fontsize=10)
axes[2].set_title(r"$\lambda=1.0$", fontsize=10)

plt.savefig("dihedral_CNCC.png", dpi=300)
plt.show()
