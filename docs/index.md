<div align="center">
    <h1>Sudoku Solver</h1>
    <p>
        An advanced Sudoku solving toolkit implementing both Constraint Propagation and a Local Search Genetic Algorithm (LSGA).
    </p>
</div>

<br/>

# 📘 Table of Contents

- [📘 Table of Contents](#-table-of-contents)
  - [🌟 About the Project](#-about-the-project)
    - [👾 Tech Stack](#-tech-stack)
    - [🎯 Features](#-features)
  - [🔧 Configuration and Environment Variables](#-configuration-and-environment-variables)
    - [Example Environment Variables](#example-environment-variables)
  - [🧰 Getting Started](#-getting-started)
    - [❗ Prerequisites](#-prerequisites)
    - [⚙ Installation](#-installation)
    - [Run Project Reference Site Locally](#run-project-reference-site-locally)
    - [🔬 Running Tests](#-running-tests)
    - [🚀 Run Locally](#-run-locally)
  - [👀 Usage](#-usage)
  - [🗺 Roadmap](#-roadmap)
  - [⚠ License](#-license)
  - [💎 Acknowledgements](#-acknowledgements)

---

## 🌟 About the Project

This Sudoku Solver project is designed to solve Sudoku puzzles using two state-of-the-art techniques:

1. A **Constraint Propagation** approach that leverages backtracking, forward-checking, and advanced heuristics such as MAC (Maintaining Arc Consistency), MRV (Most Restrictive Variable), and LCV (Least Constraining Value).
2. A **Local Search Genetic Algorithm (LSGA)** based on a novel evolutionary approach that uses column and sub-block local search to effectively balance exploration and exploitation.

Both algorithms are benchmarked on a large dataset of Sudoku puzzles, allowing detailed performance analysis in terms of speed, memory efficiency, and backtracking steps. Results from these analyses offer insights into each algorithm's strengths and weaknesses across varying levels of Sudoku difficulty.

---

### 👾 Tech Stack

<details>
    <summary>Core Languages and Libraries</summary>
        <ul>
            <li><a href="https://www.python.org/">Python 3.12</a></li>
            <li><a href="https://numpy.org/">NumPy</a> for efficient matrix operations</li>
        </ul>
</details>

### 🎯 Features

- Solves Sudoku puzzles using both Constraint Propagation and Genetic Algorithms.
- Benchmarking capabilities for evaluating performance, including tracking memory and CPU usage.
- Detailed logging and configurable settings for custom execution parameters.
- Advanced heuristics and local search techniques for enhanced algorithmic efficiency.

---

## 🔧 Configuration and Environment Variables

To customize the application, adjust variables in `config.py`.

### Example Environment Variables

```python
# Population Size
population_size = 150
# Elite Poulation Size
elite_size = 50
# Tournament Size
tournament_size = 2
# Maximum Generations Count
max_generations = 200
# PC1 or Individual Crossover Rate
individual_crossover_rate = 0.2
# PC2 or Row Crossover Rate
row_crossover_rate = 0.1
# PM1 or Swap Mutation Rate
swap_mutation_rate = 0.3
# PM2 or Reinitialization Mutation Rate
reinitialization_mutation_rate = 0.05
```

---

## 🧰 Getting Started

### ❗ Prerequisites

- **Python 3.12 or later** is required to run this project.
- We recommend using **Poetry** for dependency management. Install it globally:

```bash
pip install poetry
```

### ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/PayThePizzo/SudokuSolver.git
```

Navigate to the project directory:

```bash
cd sudoku-solver
```

Create the virtual environment:

```bash
# Root folder
python -m venv .venv
```

Activate the environment:

```bash
# Root folder (Windows)
.venv\Scripts\activate
```

Install dependencies with poetry:

```bash
(.venv) pip install poetry

(.venv) poetry install
```

Install dependencies with pip:

```bash
(.venv) pip install -r requirements.txt
```

### Run Project Reference Site Locally

You can run the project reference site to quickly consult the reference and read the comparative analysis.

Just run

```bash
mkdocs serve
```

### 🔬 Running Tests

To execute unit tests for both Constraint Propagation and LSGA modules, run:

```bash
# MAC
poetry run pyton /tests/test_sudoku_solver_MAC.py

# LSGA
poetry run pyton /tests/test_sudoku_solver_LSGA.py
```

### 🚀 Run Locally

To run the project locally, run the solver after adding a given puzzle:

```bash
# MAC
poetry run python sudoku_solver_CSP.py

# LSGA
poetry run python sudoku_solver_LSGA.py
```

---

## 👀 Usage

This toolkit can be used to analyze and solve any 9x9 Sudoku puzzle, either by specifying a puzzle directly in the code or by loading from a dataset.

---

## 🗺 Roadmap

- [x] Implement Constraint Propagation algorithm with MAC, MRV, LCV and Backtracking
- [x] Integrate Local Search Genetic Algorithm for alternative solving
- [x] Implement performance tracking and benchmarking across multiple puzzles
- [ ] Add a web interface for interactive Sudoku solving and visualization
- [ ] Improve LSGA mutation and crossover strategies
- [ ] Improve MAC with Numpy
- [ ] Fix Logger
- [ ] Add GPU support for LSGA and MAC with PyTorch or Tensorflow
- [ ] Fix Unit Tests

---

## ⚠ License

Distributed under the GNU GPL v3 License. See `LICENSE` for more information.

---

## 💎 Acknowledgements

This project draws on foundational work in AI and Sudoku-solving algorithms. Notable references include:

- The implementation of LSGA is completely inspired by the paper: [A Novel Evolutionary Algorithm With Column and Sub-Block Local Search for Sudoku Puzzles](https://ieeexplore.ieee.org/document/10015696) by Chuan Wang et al. The algorithm has been implemented to the best of our capabilities, with aim of reproducing their work. Our implementatio is **not official** by any means and is not to be considered like a distribution of the original algorithm.
- [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/) by Stuart Russell and Peter Norvig.
