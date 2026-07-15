#-----------------------------------------------------------------------------------------

import figstyle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openmm import unit

#-----------------------------------------------------------------------------------------
# INPUTS

T_lbda = 298.15 #K
T_atoms = 298.15 #K
k_input = (unit.MOLAR_GAS_CONSTANT_R).value_in_unit(unit.kilojoules_per_mole/unit.kelvin)
k_output = (unit.MOLAR_GAS_CONSTANT_R).value_in_unit(unit.kilocalories_per_mole/unit.kelvin)
eq_size = 2500 #1ns
block_size = 2500 #1ns
block_time = 0.5 #ns

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
data["time"] = (data['#"Step"'])*1e-6
data['lbda'] = 1.0 - np.abs(1.0 - data['theta (dimensionless)'])
data["UsoftLJ"] = data["UsoftLJ (kJ/mol)"]
data["ULJCoul"] = data["ULJCoul (kJ/mol)"]

files = ["../opes_bias.csv"]
dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)
data2 = pd.concat(dfs, ignore_index=True)
data2.drop(index=range(eq_size), inplace=True) 
data2.index = np.arange(0, len(data2))

nblocks = len(data) // block_size

def S22(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else (6*x**2 - 15*x + 10)*x**3)
def S22p(x):
    return 0.0 if x < 0 else (0.0 if x > 1 else 30*(x**2 - 2*x + 1)*x**2)
def S32(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else 10*x**6-24*x**5+15*x**4)
def S32p(x):
    return 0.0 if x < 0 else (0.0 if x > 1 else 6*10*x**5-5*24*x**4+4*15*x**3)
start  = dict(A=0.0, B=0.35)
finish = dict(A=1.0, B=1.0)
data['hA'] = 0.0
data['hB'] = 0.0
data['hpA'] = 0.0
data['hpB'] = 0.0
for state in range(len(data)): 
    lam = data["lbda"][state]
    xA = (lam - start['A'])/(finish['A'] - start['A'])
    xB = (lam - start['B'])/(finish['B'] - start['B'])
    data.at[state, 'hA']  = S22(xA)
    data.at[state, 'hpA'] = S22p(xA) * (1/(finish['A'] - start['A']))
    data.at[state, 'hB']  = S32(xB)
    data.at[state, 'hpB'] = S32p(xB) * (1/(finish['B'] - start['B']))
data["Up"] = data['hpA']*data["UsoftLJ"] + data['hpB']*(data["ULJCoul"] - data["UsoftLJ"])

beta_out = 1.0 / (k_output * T_lbda) 
beta = 1.0 / (k_input*T_lbda)
data['w'] = np.exp(beta * data2["opes_total_bias (kJ/mol)"])   # REWEIGHT

#-----------------------------------------------------------------------------------------
# TEMPERATURE

data['T_lbda'] = data['Extension Temperature (K)'].expanding().mean()
fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data['time'], data['T_lbda'], 'k-')
ax.plot(np.array([0, data['time'].iloc[-1]]), np.array([T_lbda, T_lbda]), 'k--')
ax.set_xlabel("time (ns)")
ax.set_ylabel(r"$T$ (K)")
ax.set_ylim(T_lbda-100, T_lbda+100)
fig.savefig("temperature_lambda.png", dpi=300)

data['T_atoms'] = data['Temperature (K)'].expanding().mean()
fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data['time'], data['T_atoms'], 'k-')
ax.plot(np.array([0, data['time'].iloc[-1]]), np.array([T_atoms, T_atoms]), 'k--')
ax.set_xlabel("time (ns)")
ax.set_ylabel(r"$T$ (K)")
ax.set_ylim(T_atoms-100, T_atoms+100)
fig.savefig("temperature_atoms.png", dpi=300)

#-----------------------------------------------------------------------------------------
# BIAS VS LAMBDA 

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0),  sharex=True, dpi=300)
ax.plot(data['lbda'], (k_output / k_input) * data2["opes_conv_bias (kJ/mol)"], 'bo', label='')
ax.plot(data['lbda'], (k_output / k_input) * data2["opes_expl_bias (kJ/mol)"], 'ro', label='')
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$U_\text{bias}$ (kcal/mol)')
fig.savefig("bias.png", dpi=300)

#-----------------------------------------------------------------------------------------
# LAMBDA VS TIME

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(data['time'], data['lbda'], 'k-', label='')
ax.set_ylabel(r'$\lambda$')
ax.set_xlabel(r'time (ns)')
fig.savefig("lambda_time.png", dpi=300)

#-----------------------------------------------------------------------------------------
# FREE ENERGY 

bins = np.linspace(0, 1, 31)  
labels = 0.5 * (bins[1:] + bins[:-1])  
labels_full = np.concatenate(([0.0], labels, [1.0]))
deltaFij = np.zeros((nblocks, labels_full.size))
deltaFprobij = np.zeros((nblocks, labels_full.size))
for i in range(nblocks):
    subset = data.iloc[: (i+1)*block_size].copy()
    subset['lambda_bin'] = pd.cut(
        subset['lbda'],
        bins=bins,
        labels=labels,
        include_lowest=True)


    # --- FREE ENERGY by probability ---
    # P0(bin) ∝ sum(w) in the bin
    # counts_w = subset.groupby('lambda_bin', observed=False)['w'].sum().reindex(labels)
    # P = counts_w / counts_w.sum()
    # F_prob = -(1.0 / beta_out) * np.log(P.values)
    # F_prob[np.isinf(F_prob)] = np.nan  # bins not visited -> NaN
    # F_prob = F_prob - F_prob[0]
    # F_prob_full = np.concatenate(([F_prob[0]], F_prob, [F_prob[-1]]))
    # deltaFprobij[i, :] = F_prob_full

    # --- FREE ENERGY by mean force ---
    #g = subset.groupby('lambda_bin', observed=False)
    #num = (g.apply(lambda x: np.sum(x['w'] * x['Up']))).reindex(labels)
    #den = (g['w'].sum()).reindex(labels)
    #mean_Up = (num / den)
    #   
    mean_Up = subset.groupby('lambda_bin', observed=False)['Up'].mean().reindex(labels)
    y = mean_Up.values
    y_full = np.concatenate(([0.0], y, [0.0]))
    dx_full = labels_full[1:] - labels_full[:-1]
    deltaF_full = np.zeros_like(labels_full, dtype=float)
    deltaF_full[1:] =  np.cumsum(0.5 * (y_full[1:] + y_full[:-1]) * dx_full)
    deltaFij[i, :] = (k_output / k_input) * deltaF_full

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(labels_full, deltaFij[-1,:], 'ko-')
#ax.plot(labels_full, deltaFprobij[-1,:], 'bo-')
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\Delta G$ (kcal/mol)')
fig.savefig("free_energy_profile.png", dpi=300)

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)
ax.plot(labels_full, (k_output / k_input) * np.concatenate(([0.0], mean_Up, [0.0])), 'ko-')
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\langle \partial U/ \partial \lambda \rangle$ (kcal/mol)')
fig.savefig("mean_force.png", dpi=300)

#-----------------------------------------------------------------------------------------
# CONVERGENCE

print(f"deltaG (all): {deltaFij[-1,-1]:.2f} kcal/mol")
#print(f"deltaG (probability): {deltaFprobij[-1,-1] - deltaFprobij[-1,0]:.2f} kcal/mol")

time = np.arange(1,int(len(deltaFij[:,-1]))+1)*block_time

fig, ax = plt.subplots(1, 1, figsize=(2.0, 2.0), dpi=300)

ax.plot(time, deltaFij[:,-1], 'k-', label=r"$\lambda$-dynamics")

plt.legend(fontsize=8,ncols=2,bbox_to_anchor=(0.5, 1.1),loc='center')
plt.xlabel(r'time (ns)')
plt.ylabel(r'$\Delta G$ (kcal/mol)')
fig.savefig("free_energy_time.png", dpi=300)

np.savez("block_deltaF.npz", deltaFij = deltaFij[:, -1], time_blocks = time[:])


