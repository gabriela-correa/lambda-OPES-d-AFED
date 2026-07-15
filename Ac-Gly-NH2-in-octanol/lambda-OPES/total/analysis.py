#-----------------------------------------------------------------------------------------

import figstyle
import numpy as np
from matplotlib import pyplot as plt

#-----------------------------------------------------------------------------------------

deltaF_mean = -13.45
deltaF_std = 0.08

data_1 = np.load("../replicate-1/post-processing/block_deltaF.npz")
deltaFij_1 = data_1["deltaFij"][:52]
time_blocks_1 = data_1["time_blocks"][:52]

data_2 = np.load("../replicate-2/post-processing/block_deltaF.npz")
deltaFij_2 = data_2["deltaFij"][:52]
time_blocks_2 = data_2["time_blocks"][:52]

data_3 = np.load("../replicate-3/post-processing/block_deltaF.npz")
deltaFij_3    = data_3["deltaFij"][:52]
time_blocks_3 = data_3["time_blocks"][:52]

deltaF = np.mean(np.vstack([deltaFij_1, deltaFij_2, deltaFij_3]), axis=0)

#-----------------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)

t0, t1 = time_blocks_3[0], time_blocks_3[-1]

# ===== λ-dynamics =====
# running estimate
dyn_traj, = ax.plot(
    time_blocks_3,
    deltaF,
    'b-',
    label=r"$\lambda$-dynamics"
)

# mean value
dyn_mean_line, = ax.plot(
    [t0, t1],
    [deltaF_mean, deltaF_mean],
    'b--'
)

# uncertainty band for λ-dynamics
dyn_band = ax.fill_between(
    time_blocks_3,
    deltaF_mean - deltaF_std,
    deltaF_mean + deltaF_std,
    color='b',
    alpha=0.15,
    edgecolor='none'
)

# ===== labels & limits =====
ax.set_ylim(-14.5,-12.5)
ax.set_xlim(0,50)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel(r'time (ns)')
ax.set_ylabel(r'$\Delta G$ (kcal/mol)')

# ===== legend (simple) =====
ax.legend(
    fontsize=8,
    loc='upper center',
    frameon=False
)

fig.savefig("free_energy.png", dpi=300)

np.savez("data.npz", deltaF=deltaF, time=time_blocks_3)


