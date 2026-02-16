# COIL_OPT — Stellarator Coil Optimization with SIMSOPT

A collection of examples and tools for optimizing electromagnetic coil designs for stellarator fusion devices, built on top of the [SIMSOPT](https://github.com/hiddenSymmetries/simsopt) framework.

## Overview

Stellarators confine plasma using carefully shaped external magnetic coils rather than relying on plasma current (as tokamaks do). Designing these coils is a complex geometric optimization problem: the coils must produce a magnetic field that closely matches a target equilibrium on the plasma boundary while satisfying engineering constraints on length, curvature, and inter-coil spacing.

**COIL_OPT** extends SIMSOPT's coil optimization capabilities with ready-to-run examples for various stellarator configurations, interactive 3D visualization, and a structured development workflow for adding new use cases.

## Features

- **Stage II coil optimization** — find coil shapes that reproduce a target normal magnetic field on a given plasma surface
- **Stochastic optimization** — account for manufacturing tolerances via systematic and statistical coil perturbations
- **Multiple stellarator configurations** — examples for QA (quasi-axisymmetric) and W7-X-like geometries
- **Interactive 3D visualization** — Plotly-based scripts and a Streamlit app for exploring coils and plasma surfaces
- **VTK output** — coil curves and surfaces exported as `.vtu` / `.vts` for use with ParaView or other tools

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [SIMSOPT](https://simsopt.readthedocs.io/) and its dependencies (including MPI support)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/COIL_OPT.git
cd COIL_OPT

# Install in development mode
pip install -e ".[dev]"

# Install SIMSOPT (if not already installed)
pip install simsopt
```

### Run an Example

```bash
cd basic_examples

# Example 1 — QA stochastic coil optimization (Landreman & Paul 2021)
python example_1_case.py

# Example 2 — W7-X Stage II optimization with virtual casing
python example_2_case.py
```

Output files (`.vtu` / `.vts`) are written to `basic_examples/output/`.

### Visualize Results

```bash
# Interactive Plotly plot (optimized coils by default)
python plotting_coils.py

# Plot initial coils instead
python plotting_coils.py init

# Streamlit app with sidebar controls
streamlit run streamlit_coils.py
```

## Examples

### Example 1 — Stochastic QA Coil Optimization

Solves a FOCUS-like Stage II problem for the quasi-axisymmetric configuration from [arXiv:2108.03711](https://arxiv.org/abs/2108.03711). The objective minimizes the mean squared normal field over stochastically perturbed coils, plus regularization penalties:

$$J = \frac{1}{2}\,\mathbb{E}\!\left[\int |B \cdot n|^2 \, ds\right] + \lambda_L \sum \text{CurveLength} + \lambda_d \,\text{MinDist} + \lambda_\kappa \sum \text{Curvature} + \lambda_{\text{msc}} \sum \text{MSC} + \lambda_a \sum \text{ArclengthVar}$$

- 4 unique coil shapes per half-field-period (nfp = 2, 16 coils total via symmetry)
- Fourier order 24 coil representation
- Gaussian perturbation model with systematic + statistical error components
- L-BFGS-B optimizer (400 iterations)

### Example 2 — W7-X Virtual Casing

Optimizes coils for a W7-X configuration at 4% average beta. Because the equilibrium is not vacuum, a **virtual casing** calculation provides the target $B_{\text{external}} \cdot n$:

$$J = \frac{1}{2} \int |B_{\text{BiotSavart}} \cdot n - B_{\text{external}} \cdot n|^2 \, ds + \lambda \sum \frac{1}{2}(\text{CurveLength} - L_0)^2$$

- 5 unique coil shapes per half-field-period (nfp = 5, 50 coils total)
- Fourier order 6 coil representation
- L-BFGS-B optimizer (500 iterations)

## Project Structure

```
COIL_OPT/
├── basic_examples/           # Runnable optimization & visualization scripts
│   ├── example_1_case.py     # Stochastic QA coil optimization
│   ├── example_2_case.py     # W7-X virtual casing optimization
│   ├── plotting_coils.py     # Plotly 3D coil viewer (pyvista backend)
│   ├── plotting_example2.py  # Plotly 3D viewer (meshio backend)
│   ├── streamlit_coils.py    # Interactive Streamlit visualization app
│   └── output/               # Generated VTU/VTS files
├── data/test_files/          # VMEC input files & equilibria for examples
├── docs/                     # Project documentation & development context
├── tests/                    # Test suite
├── scripts/                  # Development guardrail scripts
├── prompts/                  # Reusable agent prompts for phased development
├── pyproject.toml            # Build configuration
└── README.md
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| [simsopt](https://simsopt.readthedocs.io/) | Stellarator optimization framework (Biot-Savart, coil geometry, objectives) |
| [scipy](https://scipy.org/) | L-BFGS-B and other optimizers |
| [numpy](https://numpy.org/) | Array operations |
| [plotly](https://plotly.com/python/) | Interactive 3D visualization |
| [pyvista](https://docs.pyvista.org/) | VTK file I/O and mesh processing |
| [streamlit](https://streamlit.io/) | Web-based interactive dashboard |

## Development Workflow

This repository uses a **phased development framework** for structured, agent-assisted development:

1. **IDEA** — problem statement and use cases
2. **SPEC** — formal requirements with traceable IDs (`REQ-*`)
3. **DEV PLAN** — task breakdown and execution plan
4. **TESTS** — pytest tests mapped to requirements
5. **IMPLEMENTATION** — code that satisfies the specs

See [AGENT_README.md](AGENT_README.md) for full framework documentation, and [docs/context.md](docs/context.md) for the current project status.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-configuration`)
3. Follow the phased workflow described in [AGENT_README.md](AGENT_README.md)
4. Ensure tests pass: `pytest -q`
5. Open a pull request

## License

See the repository license file for details.

## References

- Landreman, M. & Paul, E. (2021). *Magnetic fields with precise quasisymmetry for plasma confinement.* [arXiv:2108.03711](https://arxiv.org/abs/2108.03711)
- [SIMSOPT Documentation](https://simsopt.readthedocs.io/)
- [SIMSOPT GitHub Repository](https://github.com/hiddenSymmetries/simsopt)