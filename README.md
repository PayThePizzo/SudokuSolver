
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
    <li><a href="https://www.python.org/">Python</a></li>
    <li><a href="https://numpy.org/">NumPy</a> for efficient matrix operations</li>
    <li><a href="https://docs.python.org/3/library/logging.html">Python Logging</a> for application-wide logging and debugging</li>
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

```plaintext
LOG_LEVEL=INFO
MAX_GENERATIONS=200
POPULATION_SIZE=150
ELITE_SIZE=50
TOURNAMENT_SIZE=2
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

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/sudoku-solver.git
    ```
2. Navigate to the project directory:
    ```bash
    cd sudoku-solver
    ```
3. Install dependencies:
    ```bash
    poetry install
    ```

### 🔬 Running Tests

To execute unit tests for both Constraint Propagation and LSGA modules, run:
```bash
poetry run pytest
```

### 🚀 Run Locally

To run the project locally:
1. Run the solver for a given puzzle:
    ```bash
    poetry run python sudoku_solver.py
    ```
2. You can switch between Constraint Propagation or LSGA mode by modifying `config.py` or using environment variables.

---

## 👀 Usage

This toolkit can be used to analyze and solve any 9x9 Sudoku puzzle, either by specifying a puzzle directly in the code or by loading from a dataset. See `example_usage.py` for sample usage of both the Constraint Propagation and LSGA-based solvers.

---

## 🗺 Roadmap

- [x] Implement Constraint Propagation algorithm with MAC, MRV, and LCV
- [x] Integrate Local Search Genetic Algorithm for alternative solving
- [x] Implement performance tracking and benchmarking across multiple puzzles
- [ ] Add a web interface for interactive Sudoku solving and visualization
- [ ] Improve LSGA mutation and crossover strategies

---

## ⚠ License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

## 💎 Acknowledgements

This project draws on foundational work in AI and Sudoku-solving algorithms. Notable references include:

- [A Novel Evolutionary Algorithm With Column and Sub-Block Local Search for Sudoku Puzzles](https://ieeexplore.ieee.org/document/10015696) by Chuan Wang et al.
- [Artificial Intelligence: A Modern Approach](https://aima.cs.berkeley.edu/) by Stuart Russell and Peter Norvig.
```

This README template provides a comprehensive overview, setup instructions, and resource links, along with specific project details relevant to your Sudoku Solver implementation.