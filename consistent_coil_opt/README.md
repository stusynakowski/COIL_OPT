# Consistent Coil Optimization & Visualization

This directory contains resources for two-stage coil optimization using PyTorch and interactive visualization of the results.

## Prerequisites

Ensure you have Python 3.10+ installed. It is recommended to use a virtual environment or Conda environment.

## Installation

1.  **Install Dependencies**

    You will need `simsopt`, `torch`, `streamlit`, `plotly`, `pyvista`, `matplotlib`, and `numpy`.

    ```bash
    pip install simsopt torch torchvision torchaudio streamlit plotly pyvista matplotlib numpy
    ```

    *Note: `simsopt` may require specific installation steps depending on your OS. Refer to the [SIMSOPT documentation](https://github.com/hiddenSymmetries/simsopt) if you encounter issues.*

2.  **Verify Installation**

    Run the following to ensure the environment is set up correctly:

    ```bash
    python -c "import simsopt; import torch; import streamlit; print('Dependencies installed successfully')"
    ```

## Running the Optimization

The optimization process is contained within a Jupyter Notebook.

1.  Navigate to this directory:
    
    ```bash
    cd consistent_coil_opt
    ```

2.  Launch Jupyter Notebook or Lab:

    ```bash
    jupyter notebook stage_two_optimization_pytorch_rotated.ipynb
    ```

3.  Run all cells in the notebook. This will:
    - Load the initial coil configuration.
    - Perform optimization using PyTorch LBFGS.
    - Save the resulting coil data (VTU/VTS files) to the `output_pytorch_rotated` directory.

## Running the Visualization

After running the optimization (or if you have existing results in `output_pytorch_rotated`), you can visualize the coils and plasma surface interactively.

1.  Run the Streamlit app:

    ```bash
    streamlit run visualization_v2_streamlit.py
    ```

2.  A new tab should open in your default web browser displaying the 3D visualization tool. You can toggle between "Initial" and "Optimized" views to compare the results.

## File Structure

- `stage_two_optimization_pytorch_rotated.ipynb`: Main optimization notebook using PyTorch.
- `visualization_v2_streamlit.py`: Streamlit application for 3D visualization.
- `output_pytorch_rotated/`: Directory where output files (VTU, VTS, JSON) are saved.
- `main_pipeline.py`: (Optional/Legacy) Script for similar optimization tasks.
