#-----------------------------------------------------------------------------------------

import figstyle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#-----------------------------------------------------------------------------------------

deltaF_mean = -7.01
deltaF_std = 0.06

df1 = pd.read_csv("../replicate-1/DeltaG_vs_time.csv")
df2 = pd.read_csv("../replicate-2/DeltaG_vs_time.csv")
df3 = pd.read_csv("../replicate-3/DeltaG_vs_time.csv")

deltaFij_1 = df1["DeltaG_kcal_per_mol"].values
time_1     = (df1["time_ns"].values)*21

deltaFij_2 = df2["DeltaG_kcal_per_mol"].values
time_2     = (df2["time_ns"].values)*21

deltaFij_3 = df3["DeltaG_kcal_per_mol"].values
time_3     = (df3["time_ns"].values)*21

deltaF = np.mean(np.vstack([deltaFij_1, deltaFij_2, deltaFij_3]), axis=0) 

#-----------------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)

t0, t1 = time_1[0], time_1[-1]

dyn_traj, = ax.plot(time_1, deltaF, 'b-')

dyn_mean_line, = ax.plot([t0, t1], [deltaF_mean, deltaF_mean], 'b--')

dyn_band = ax.fill_between(
    time_1,
    deltaF_mean - deltaF_std,
    deltaF_mean + deltaF_std,
    color='b',
    alpha=0.15,
    edgecolor='none'
)

ax.set_ylim(-8.5,-5.5)
ax.set_xlabel(r'time (ns)')
ax.set_ylabel(r'$\Delta G$ (kcal/mol)')
ax.set_xlim(0, 20)
ax.set_xticks([0, 5, 10, 15, 20])

fig.savefig("free_energy_REX.png", dpi=300)

np.savez("REX.npz", deltaF=deltaF, time=time_1)



