
#conda activate ommt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeAnalyzer
from pymbar import MBAR
from openmm import unit

# -----------------------
# USER SETTINGS
# -----------------------
ncfile = "rex_lambda_cv.nc"

T = 298.15 * unit.kelvin
kT_kcal = (unit.MOLAR_GAS_CONSTANT_R * T).value_in_unit(unit.kilocalories_per_mole)

t0 = 24          # burn-in em "iterations"
g  = 1             # stride (1 = usa tudo após t0)
iter_ps = 1.0      # <<<<< 1 ps por iteração (informado por você)

step_eval = 24    # calcula um ponto de MBAR a cada N iterações (aumente se ficar lento)
state_i = 0        # estado inicial
# state_j = -1     # estado final (deixo definido abaixo como K-1)

# -----------------------
# HELPERS
# -----------------------
def mbar_cumulative_deltaG(u_kln, ana, t0, g, i_max, state_i, state_j, kT):
    """
    MBAR cumulativo usando amostras de t0 até i_max (exclusive), com stride g.
    Retorna (DeltaG, dDeltaG) em unidades de energia (kT já multiplicado).
    """
    K = ana.n_states
    Ntot = u_kln.shape[2]

    i_max = int(min(max(i_max, 0), Ntot))
    if i_max <= t0:
        return np.nan, np.nan

    # máscara temporal sobre N (iterações)
    cols = np.zeros(Ntot, dtype=bool)
    cols[t0:i_max:g] = True
    n_keep = int(cols.sum())

    # regra prática: precisa de estatística mínima
    if n_keep < max(50, K + 1):
        return np.nan, np.nan

    # corta e re-formata
    u_kln_cut = u_kln[:, :, cols]                      # (K, K, n_keep)
    u_kn_cut  = ana.reformat_energies_for_mbar(u_kln_cut)

    # assumindo número igual de amostras por estado (padrão do reformat)
    N_k = np.full(K, n_keep, dtype=int)

    mbar = MBAR(u_kn_cut, N_k, verbose=False)
    out = mbar.compute_free_energy_differences()

    DeltaG  = out["Delta_f"][state_i, state_j]  * kT
    dDeltaG = out["dDelta_f"][state_i, state_j] * kT
    return float(DeltaG), float(dDeltaG)

# -----------------------
# MAIN
# -----------------------
rep = MultiStateReporter(ncfile, open_mode="r")
ana = ReplicaExchangeAnalyzer(rep)

u_kln, _, _, _ = ana.read_energies()   # u_kln: (K, K, N)
K = ana.n_states
Ntot = u_kln.shape[2]

state_j = K - 1

# pontos de avaliação cumulativa
i_list = np.arange(t0 + step_eval, Ntot + 1, step_eval, dtype=int)

DG = np.empty(len(i_list), dtype=float)
dDG = np.empty(len(i_list), dtype=float)

for idx, i_max in enumerate(i_list):
    dg, ddg = mbar_cumulative_deltaG(
        u_kln=u_kln,
        ana=ana,
        t0=t0,
        g=g,
        i_max=i_max,
        state_i=state_i,
        state_j=state_j,
        kT=kT_kcal
    )
    DG[idx]  = dg
    dDG[idx] = ddg
    print(f"i_max={i_max:7d}  t={(i_max*iter_ps/1000):8.3f} ns  ΔG={dg:9.3f} ± {ddg:7.3f} kcal/mol")

# eixo do tempo em ns (1 ps/iter = 1e-3 ns/iter)
time_ns = i_list * (iter_ps / 1000.0)

# -----------------------
# PLOT
# -----------------------
plt.figure()
plt.errorbar(time_ns, DG, yerr=dDG, fmt='o-', capsize=3)
plt.xlabel("Simulation time (ns)")
plt.ylabel(r"$\Delta G$ (kcal/mol)")
plt.tight_layout()
plt.savefig("free_energy.png", dpi=300)

# monta o DataFrame
df = pd.DataFrame({
    "time_ns": time_ns,
    "DeltaG_kcal_per_mol": DG,
    "dDeltaG_kcal_per_mol": dDG
})

# salva em CSV
df.to_csv(
    "DeltaG_vs_time.csv",
    index=False
)
