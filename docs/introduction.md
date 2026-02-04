# Introduction

## Background
We are engineering a stellarator (a "fancy tokamak"), where the configuration of the magnetic field is crucial for creating the right conditions for fusion. Unlike tokamaks, stellarators rely entirely on external coils to induce the twisting magnetic field required for plasma confinement.

## Problem Statement
The technical problem we are solving is determining the optimal coil design. We are trying to induce a specific target magnetic field using a set of coils, which presents a complex geometric optimization problem.

This project addresses:
1.  **Modeling of Magnetic Fields:** Simulating the field induced by current loops.
2.  **Coil Design:** Defining the parameters that control coil geometry.
3.  **Optimization:** Strategies to minimize the difference between the induced field and the target field, subject to manufacturing and physical constraints.

## Users / stakeholders
- Stellarator Engineers
- Plasma Physicists

## Use Cases
- **UC-001:** Optimize coil geometry to match a target magnetic surface.
- **UC-002:** Evaluate magnetic field properties for a given coil configuration.

## Constraints
- **Language:** Python
- **Testing:** pytest
- **Packaging:** src-layout
- **Non-goals:** Plasma physics simulation (focus is strictly on coil/field geometry).

## Success Criteria
- Successfully implementing an optimization loop that converges on a coil design producing the required magnetic field within tolerance.
