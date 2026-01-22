**A Conceptual Model for Ligand Ranking in Drug Design based on Information Vector Theory and ODE Dynamics.**

This repository contains the reference implementation of the **Information Force Framework**, as described in the manuscript submitted. The framework utilizes a vectorized ODE solver to simulate protein-ligand binding dynamics in a semantic property space, offering ultra-fast ($O(N)$) and interpretable ligand ranking.

## 🚀 Key Features

- **High-Throughput Performance:** Processes 10,000 ligands in approximately **5 milliseconds**.
- **Physics-Based Interpretability:** Uses a damped harmonic oscillator model ($m, c, k$) to assess ligand quality.
- **Fully Vectorized:** Implemented in Python using NumPy broadcasting to ensure linear time complexity $O(N)$ without iterative loops.
- **Reproducible:** Includes fixed random seed initialization to guarantee identical results across runs.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher (Python 3.12 recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/SlawomirWisniewski73/Conceptual-Model-for-Ligand-Ranking-in-Drug-Design.git]
   cd Information-Force-Framework


2. Install required dependencies:
   ```bash
   pip install -r requirements.txt2. Install required dependencies: Bash pip install -r requirements.txt




📊 Reproducing the Paper Results

To verify the findings presented in the manuscript (Figures 1-4 and the statistical analysis of 10,000 ligands), run the reproduction script: Bash python reproduce_results.py
This script will perform the following actions:
- Generate a synthetic dataset of 10,000 ligands based on pharmaceutical property ranges.
- Initialize the VectorizedLigandSolver with the default biological weights.
- Compute ranking metrics (Final Distance, VCF) for the entire population.
- Save the full results dataset to ligands_10k_supplementary.csv.
- Generate and save the figures used in the publication:
  - fig1_heatmap.png (Parameter Sensitivity)
  - top5_analysis.png (Lead Optimization Divergence)
  - fig3_hist.png (Score Distribution)
  - fig4_sensitivity.png (Weight Analysis)

📂 Repository Structure
FileDescription
- solver_vector.py --> Core Engine. Contains the VectorizedLigandSolver class and physics logic.
- reproduce_results.py --> Experiment Script. Generates data, runs the solver, and creates plots.
- requirements.txt --> List of Python dependencies (numpy, pandas, matplotlib, seaborn).
- ligands_10k_supplementary.csv --> Output file containing the generated dataset and results.

🔗 Data Availability

The raw dataset and generated metrics are also archived on Figshare to ensure long-term accessibility: https://doi.org/10.6084/m9.figshare.31129420

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

📚 Citation

If you use this code or methodology in your research, please cite the following paper:
