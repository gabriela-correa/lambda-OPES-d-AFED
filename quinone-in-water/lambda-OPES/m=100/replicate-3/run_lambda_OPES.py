
#-----------------------------------------------------------------------------------------
# IMPORT

from openmm.app import *
from openmm import *
from openmm.unit import *
import nonbondedslicing as nbs
import random
import pandas as pd
import itertools
from collections import namedtuple

import openxps as xps
import cvpack

#-----------------------------------------------------------------------------------------
# INPUT

seed = random.SystemRandom().randint(0, 2**31)

platform = Platform.getPlatformByName('CUDA')
properties = {"Precision": "mixed"}

start  = dict(A=0.0, B=0.35) 
finish = dict(A=1.0, B=1.0)  

dt = 1*femtoseconds 
production_time = 25*nanoseconds
sampling_interval = 200*femtoseconds
barostat_interval = 25*femtoseconds

temp = 298.15*kelvin 
pressure = 1*atmosphere 
rcut = 12*angstroms
rswitch = 11*angstroms
gamma = 10/picoseconds

tau_lbda = 40*femtoseconds
T_lbda = 298.15*kelvin
mass_lbda = 100*dalton*(nanometer)**2

res_name_solute = 'MOL'

production_nsteps = round(production_time/dt)
report_interval = round(sampling_interval/dt)
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
# COMBINING THE FORCES

cv1 = CustomCVForce("UsoftLJ")
cv1.addCollectiveVariable("UsoftLJ", force_UsoftLJ)
UsoftLJ = cvpack.OpenMMForceWrapper(cv1, kilojoules_per_mole, name="UsoftLJ") 

cv2 = CustomCVForce("ULJCoul")
cv2.addCollectiveVariable("ULJCoul", force_ULJCoul)
ULJCoul = cvpack.OpenMMForceWrapper(cv2, kilojoules_per_mole, name="ULJCoul") 

U = "hA*UsoftLJ + hB*(ULJCoul - UsoftLJ)"
U += f"; hA=(step(xA)-step(xA-1))*(6*xA^2-15*xA+10)*xA^3+step(xA-1)"
U += f"; xA=( theta*(1-step(theta-1)) + (2-theta)*step(theta-1) -{start['A']})/{finish['A']-start['A']}"
U += f"; hB=(step(xB)-step(xB-1))*(10*xB^2-24*xB+15)*xB^4+step(xB-1)"
U += f"; xB=( theta*(1-step(theta-1)) + (2-theta)*step(theta-1) -{start['B']})/{finish['B']-start['B']}"

theta_dv = xps.DynamicalVariable(name="theta", unit=dimensionless, mass=mass_lbda, bounds=xps.PeriodicBounds(0.0, 2.0, dimensionless))
coupling = xps.CollectiveVariableCoupling(f"{U}", [UsoftLJ, ULJCoul], [theta_dv])

#-----------------------------------------------------------------------------------------
# THERMOSTAT + BAROSTAT

system.addForce(MonteCarloBarostat(pressure, temp, barostat_freq))
physical = LangevinMiddleIntegrator(temp, gamma, dt)
extension = xps.integrators.RegulatedNHLIntegrator(temperature=T_lbda, timeConstant=tau_lbda, stepSize=dt, forceFirst=True)

#-----------------------------------------------------------------------------------------
# SIMULATION

ext_system = xps.ExtendedSpaceSystem(system, coupling)

theta_bias_var = xps.ExtendedSpaceBiasVariable(dynamical_variable=theta_dv, sigma=0.10*dimensionless, grid_width=101)

opes_conv = xps.ExtendedSpaceOPES(system=ext_system,
                                  variables=[theta_bias_var],
                                  temperature=T_lbda,
                                  barrier=126.0*kilojoules_per_mole,   
                                  frequency=400,
                                  biasFactor=50.8,
                                  exploreMode=False
                                  )

opes_expl = xps.ExtendedSpaceOPES(system=ext_system,
                                  variables=[theta_bias_var],
                                  temperature=T_lbda,
                                  barrier=21.0*kilojoules_per_mole,  
                                  frequency=400,
                                  biasFactor=8.5,
                                  exploreMode=True
                                  )

simulation = xps.ExtendedSpaceSimulation(top.topology,
                                         ext_system,
                                         xps.LockstepIntegrator(physical, extension),
                                         platform,
                                         properties
                                         )

#-----------------------------------------------------------------------------------------
# INITIAL CONDITION

simulation.context.setPositions(gro.positions)
simulation.context.setVelocitiesToTemperature(temp, seed)

simulation.context.setDynamicalVariableValues([0.0])
simulation.context.setDynamicalVariableVelocitiesToTemperature(T_lbda, seed)

#-----------------------------------------------------------------------------------------
# REPORTER

reporter1 = cvpack.reporting.StateDataReporter('output.csv', 
                                               report_interval, 
                                               step=True, 
                                               density=True, 
                                               kineticEnergy=True, 
                                               temperature=True, 
                                               writers=[xps.ExtensionWriter(kinetic=True, temperature=True, 
                                               dynamical_variables=True, collective_variables=True)]
                                               )
reporter2 = xps.OPESBiasReporter('opes_bias.csv',
                                 report_interval,
                                 separator=',',
                                 step=True,
                                 opesSamplers=[("conv", opes_conv._opes), ("expl", opes_expl._opes)])
simulation.reporters.append(reporter1)
simulation.reporters.append(reporter2)
#simulation.reporters.append(PDBReporter("traj.pdb", report_interval*100)) 

#-----------------------------------------------------------------------------------------
# RUN

multi_opes = xps.MultiOPESSampler([opes_conv._opes, opes_expl._opes])
multi_opes.step(simulation, production_nsteps)
simulation.saveState("continue.xml")


