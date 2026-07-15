# Electronic Supporting Information Repository

This repository contains the supporting files associated with the article:

**Integrated Alchemical and Conformational Enhanced Sampling for Solvation Free Energy Calculations**

## Overview

The repository provides input files, simulation scripts, selected simulation outputs, and post-processing analyses for representative calculations involving:

- **N-acetylglycinamide (Ac-Gly-NH₂) in octanol**;
- **quinone in water**.

The sampling protocols include λ-AFED, λ-OPES-Explore, λ-OPES, λ-OPES-d-AFED, and Hamiltonian replica exchange (REX). Calculations are organized into three independent replicates.

## Repository structure

```text
├── Ac-Gly-NH2-in-octanol/
│   ├── lambda-OPES-AFED/
│   └── lambda-OPES/
│
├── quinone-in-water/
│   ├── lambda-AFED/
│   ├── lambda-REX/
│   ├── lambda-OPES-Explore/
│   └── lambda-OPES/
│
├── gro-files/
├── top-files/
└── README.md
```

## Directory descriptions

### Ac-Gly-NH2-in-octanol

Contains simulations and analyses for N-acetylglycinamide in octanol using:

- lambda-OPES-AFED: λ-OPES-d-AFED calculations;
- lambda-OPES: λ-OPES calculations without the additional d-AFED coordinate.

### quinone-in-water

Contains simulations and analyses for quinone in water using:

- lambda-AFED: λ-AFED calculations;
- lambda-REX: HREX reference calculations;
- lambda-OPES-Explore: using an OPES-Explore bias;
- lambda-OPES: using the complete λ-OPES protocol.

Selected quinone calculations also examine the effect of the fictitious mass assigned to the λ coordinate.

### gro-files

Contains molecular structures and initial configurations.

### top-files

Contains molecular topologies, force field parameters, and solvent definitions.
