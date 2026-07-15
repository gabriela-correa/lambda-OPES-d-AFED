
from openmm.app import *
from openmm import *
from openmm.unit import *
import nonbondedslicing as nbs
import random
import pandas as pd
from copy import deepcopy
import itertools
from collections import namedtuple
import numpy as np
from openmmtools import states, mcmc, multistate, cache
import os
from mpi4py import MPI
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

#-----------------------------------------------------------------------------------------
# INPUTS

dt = 1*femtoseconds 
production_time = 5*nanoseconds
barostat_interval = 25*femtoseconds
restart_interval = 50

temp = 298.15*kelvin 
pressure = 1*atmosphere 
gamma = 10/picoseconds 
rcut = 12*angstroms
rswitch = 11*angstroms

res_name_solute = 'MOL'

lams = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

start  = dict(A=0.0, B=0.35) #initial lambda
finish = dict(A=1.0, B=1.0)  #final lambda

iterations = round(production_time/dt/1000)
barostat_freq = round(barostat_interval/dt)

#-----------------------------------------------------------------------------------------
# TOPOLOGY AND MAIN SYSTEM

gro = app.GromacsGroFile('../mol.gro')
top = app.GromacsTopFile('../mol.top',
    			periodicBoxVectors=gro.getPeriodicBoxVectors())

system = top.createSystem(
        nonbondedCutoff=rcut,
        switchDistance=rswitch,
        nonbondedMethod=PME,
        constraints=HBonds,
        rigidWater=True,
        removeCMMotion=False
        )

solute_atoms = [atom.index for atom in top.topology.atoms() if atom.residue.name == res_name_solute]
solvent_atoms = [atom.index for atom in top.topology.atoms() if atom.index not in solute_atoms]
nbforce = next(filter(lambda f: isinstance(f, NonbondedForce), system.getForces()))

_ParamTuple = namedtuple('_ParamTuple', 'charge sigma epsilon')
parameters = []
for i in range(nbforce.getNumParticles()):
    charge, sigma, epsilon = [p/p.unit for p in nbforce.getParticleParameters(i)]
    parameters.append(_ParamTuple(charge, sigma if epsilon != 0.0 else 1.0, epsilon))

#-----------------------------------------------------------------------------------------
# SOLUTE-SOLVENT TOTAL FORCE

force_ULJCoul = nbs.SlicedNonbondedForce(nbforce,2)
for index in set(solute_atoms):
    force_ULJCoul.setParticleSubset(index, 0)
for index in set(solvent_atoms):
    force_ULJCoul.setParticleSubset(index, 1)
force_ULJCoul.addGlobalParameter("param2", 0)
force_ULJCoul.addScalingParameter("param2", 0, 0, True, True)
force_ULJCoul.addScalingParameter("param2", 1, 1, True, True)

#-----------------------------------------------------------------------------------------
# SOLUTE-SOLVENT SOFT LJ FORCE

u_LJ = '4/x^12-4/x^6'
u_cap = '(-4340*x^6+10944*x^5-7200*x^4+596)/5'
definitions = ['x=r/sigma', 'sigma=(sigma1+sigma2)/2', 'epsilon=sqrt(epsilon1*epsilon2)']
expr = [f'epsilon*select(step(1-x),{u_cap},{u_LJ})'] + definitions
force_UsoftLJ = CustomNonbondedForce(';'.join(expr))
force_UsoftLJ.addPerParticleParameter('sigma')
force_UsoftLJ.addPerParticleParameter('epsilon')
for index in range(nbforce.getNumParticles()):
    _, sigma, epsilon = nbforce.getParticleParameters(index)
    force_UsoftLJ.addParticle([sigma, epsilon])
force_UsoftLJ.setUseLongRangeCorrection(nbforce.getUseDispersionCorrection())
force_UsoftLJ.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
force_UsoftLJ.setCutoffDistance(nbforce.getCutoffDistance())
force_UsoftLJ.setUseSwitchingFunction(nbforce.getUseSwitchingFunction())
force_UsoftLJ.setSwitchingDistance(nbforce.getSwitchingDistance())
force_UsoftLJ.addInteractionGroup(set(solute_atoms), set(solvent_atoms))

#-----------------------------------------------------------------------------------------
# REMOVING SOLUTE-SOLVENT TOTAL FORCE OF THE MAIN SYSTEM 

# including all solute-solute LJ + Coul interactions as Exceptions
internal_exception_pairs = []
for index in range(nbforce.getNumExceptions()):
    i, j, _, _, _ = nbforce.getExceptionParameters(index)
    i_in_group, j_in_group = i in set(solute_atoms), j in set(solute_atoms)
    if i_in_group and j_in_group:
        internal_exception_pairs.append(set([i, j]))
for i, j in itertools.combinations(set(solute_atoms), 2):
    if set([i, j]) not in internal_exception_pairs:
        chargeprod = parameters[i].charge*parameters[j].charge
        sigma = (parameters[i].sigma + parameters[j].sigma)/2
        epsilon = sqrt(parameters[i].epsilon*parameters[j].epsilon)
        nbforce.addException(i, j, chargeprod, sigma, epsilon)
# removing all solute-solvent LJ + Coul interactions
for index in set(solute_atoms):
    nbforce.setParticleParameters(index, 0.0, 1.0, 0.0)
    
#-----------------------------------------------------------------------------------------
# COMBINE THE FORCES

U = "hA*UsoftLJ + hB*(ULJCoul - UsoftLJ)"
U += f"; hA=(step(xA)-step(xA-1))*(6*xA^2-15*xA+10)*xA^3+step(xA-1)"
U += f"; xA=(lbda-{start['A']})/{finish['A']-start['A']}"
U += f"; hB=(step(xB)-step(xB-1))*(10*xB^2-24*xB+15)*xB^4+step(xB-1)"
U += f"; xB=(lbda-{start['B']})/{finish['B']-start['B']}"
cv = CustomCVForce("")
cv.addCollectiveVariable("UsoftLJ", force_UsoftLJ)
cv.addCollectiveVariable("ULJCoul", force_ULJCoul)
cv.addGlobalParameter("lbda", 1.0)
cv.setEnergyFunction(U)
system.addForce(cv)

#-----------------------------------------------------------------------------------------
# BAROSTAT

system.addForce(MonteCarloBarostat(pressure, temp, barostat_freq))
system.addForce(CMMotionRemover())

#-----------------------------------------------------------------------------------------
# REPLICA EXCHANGE

platform = Platform.getPlatformByName("CUDA")
props = {"DeviceIndex": "0", "Precision": "mixed"}

thermo = states.ThermodynamicState(system=system, temperature=temp, pressure=pressure)

class LambdaState(states.GlobalParameterState):
    lbda = states.GlobalParameterState.GlobalParameter("lbda", standard_value=1.0)

compound_states = []
for lam in lams:
    comp = states.CompoundThermodynamicState(thermo, composable_states=[LambdaState(lbda=lam)])
    compound_states.append(comp)

move = mcmc.LangevinDynamicsMove(timestep=dt, collision_rate=gamma, n_steps=1000, reassign_velocities=False, n_restart_attempts=50) # 1000 fs = 1 ps per replicate

simulation = multistate.ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=iterations, replica_mixing_scheme="swap-neighbors") # 1000 * 1 ps = 1 ns per replicate

reporter = multistate.MultiStateReporter("rex_lambda_cv.nc", open_mode=None, checkpoint_interval=restart_interval, position_interval=50, velocity_interval=0)

energy_cache = cache.ContextCache(platform=platform, platform_properties=props,
                                  capacity=21, time_to_live=None)
sampler_cache = cache.ContextCache(platform=platform, platform_properties=props,
                                  capacity=21, time_to_live=None)
simulation.energy_context_cache = energy_cache
simulation.sampler_context_cache = sampler_cache

simulation.create(
    thermodynamic_states=compound_states,
    sampler_states=states.SamplerState(positions=gro.getPositions(), box_vectors=gro.getPeriodicBoxVectors()),
    storage=reporter
)

simulation.equilibrate(2)
simulation.run()





