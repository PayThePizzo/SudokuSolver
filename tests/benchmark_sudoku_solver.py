"""Benchmark module.

This module can be used to run the solving algorithms
and record the results in specific datasets for later
analysis.
"""

import sys
import os
import threading
import psutil
import time
import pandas as pd
import numpy as np
import glob
from memory_profiler import profile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import resdir, df_path
from src.sudoku_solver_CSP import SudokuSolverMAC
from src.sudoku_solver_LSGA import SudokuSolverLSGA
from src.utils import string_to_sudoku_board, is_valid_solution


def get_random_df_sample(df, sample_size: int = 50, min_clues: int = 17, max_clues: int = 80):
    """Return a sample of the pandas dataframe with the provided clues.

    Args:
        df (pd.DataFrame): DataFrame containing Sudoku puzzles with columns
            'quizzes', 'solutions', 'clue_numbers'.
        sample_size (int): The number of Sudoku puzzles to sample.
        min_clues (int): Minimum number of clues for the sampled puzzles.
        max_clues (int): Maximum number of clues for the sampled puzzles.
    """
    if not (0 < sample_size):
        raise ValueError(
            'Sample size cannot be a 0 or less.'
        )

    if not (16 < min_clues < max_clues):
        raise ValueError(
            'Minimum number of clues must be between 17 and max_clues.'
        )

    if not (min_clues < max_clues < 81):
        raise ValueError(
            'Maximum number of clues must be between min_clues and 80.'
        )

    # Filter the dataset based on requested clue numbers
    filtered_df = df[
        (df['clue_numbers'] >= min_clues) &
        (df['clue_numbers'] <= max_clues)
    ]

    # Randomly sample from the filtered dataset
    sample = filtered_df.sample(n=sample_size, random_state=42)

    return sample


def sample_cpu_memory_usage(process, cpu_samples, rss_samples, vms_samples, solver, interval=0.05):
    """Thread function to sample CPU and memory usage at regular intervals."""
    while solver.is_solving:  # Only sample while the solver is running
        # CPU Percentage since last call
        cpu_samples.append(process.cpu_percent(interval))
        mem_info = process.memory_info()
        rss_samples.append(mem_info.rss)
        vms_samples.append(mem_info.vms)
        time.sleep(interval)

    pass


def get_sample_statistics(sample):
    """Return min, mean, max values in sample."""
    if sample:

        return np.min(sample), int(np.mean(sample)), np.max(sample)
    else:

        return 0.0, 0.0, 0.0


def benchmark_solver(sample, solver_class, results_path):
    """Perform benchmark of the provided Sudoku solver class on a sample of Sudoku puzzles.

    This function also measures CPU and memory usage during each puzzle-solving iteration.

    Args:
        sample (pd.DataFrame): DataFrame containing Sudoku puzzles with columns
            'quizzes', 'solutions', 'clue_numbers'.
        solver_class: The class to benchmark (e.g., SudokuSolverMAC).
        results_path (str): The path where results will be saved.
    """
    results = []

    for index, row in sample.iterrows():
        # Prepare clues, puzzle and solution
        clues = int(row['clue_numbers'])
        puzzle = string_to_sudoku_board(puzzle=row['quizzes'])
        solution = np.array(string_to_sudoku_board(puzzle=row['solutions']))

        # Initialize CPU, RSS, VMS tracking
        cpu_samples = []
        rss_samples = []
        vms_samples = []

        process = psutil.Process()

        # Initialize solver and track time
        initialization_time = time.perf_counter()
        solver = solver_class(puzzle)
        initialization_time = (
            time.perf_counter() - initialization_time
        )

        mem_info = process.memory_info()
        base_rss_usage = mem_info.rss
        base_vms_usage = mem_info.vms

        # Initialize sampling thread
        solver.is_solving = True
        sampling_thread = threading.Thread(
            target=sample_cpu_memory_usage,
            args=(
                process, cpu_samples, rss_samples, vms_samples, solver
            )
        )

        # Start sampling
        sampling_thread.start()
        # Start Execution Time
        execution_time = time.perf_counter()

        # Solve puzzle and stop sampling
        proposed_solution = solver.solve()
        solver.is_solving = False

        # Stop Execution Time
        execution_time = time.perf_counter() - execution_time
        # Wait for sampling to finish
        sampling_thread.join()

        # CPU Precentage Statistics
        min_cpu_usage, avg_cpu_usage, max_cpu_usage = get_sample_statistics(cpu_samples)
        # Resident Set Size Statistics
        min_rss_usage, avg_rss_usage, max_rss_usage = get_sample_statistics(rss_samples)
        # Virtual Memory Size Statistics
        min_vms_usage, avg_vms_usage, max_vms_usage = get_sample_statistics(vms_samples)

        # Initialize flags for solving validity
        is_completed = solver.solved
        is_valid, is_match = False, False

        # Check if the solver completed the puzzle correctly
        if is_completed:
            # Update format of solution
            if isinstance(proposed_solution, bool):
                # MAC solution
                proposed_solution = np.array(solver.grid)
            else:
                # LSGA solution
                proposed_solution = proposed_solution[0]

            # Check if the puzzle is valid
            is_valid = is_valid_solution(proposed_solution)

            if is_valid:
                # Check if the puzzle obtained the original solution
                is_match = np.array_equal(solution, proposed_solution)

        # Record benchmark statistics
        benchmark_statistics = {
            'index': index,
            'clues': clues,
            'initialization_time_sec': round(initialization_time, 4),
            'execution_time_sec': round(execution_time, 4),
            'min_cpu_usage_percentage': round(min_cpu_usage, 2),
            'avg_cpu_usage_percentage': round(avg_cpu_usage, 2),
            'max_cpu_usage_percentage': round(max_cpu_usage, 2),
            'base_rss_usage_bytes': base_rss_usage,
            'min_rss_inc_bytes': min_rss_usage - base_rss_usage,
            'avg_rss_inc_bytes': avg_rss_usage - base_rss_usage,
            'max_rss_inc_bytes': max_rss_usage - base_rss_usage,
            'base_vms_usage_bytes': base_vms_usage,
            'min_vms_inc_bytes': min_vms_usage - base_vms_usage,
            'avg_vms_inc_bytes': avg_vms_usage - base_vms_usage,
            'max_vms_inc_bytes': max_vms_usage - base_vms_usage,
            'success': is_completed,
            'valid': is_valid,
            'match': is_match,
        }

        # Merge dictionaries
        combined_statistics = (
            benchmark_statistics | solver.get_statistics()
        )

        # Record results
        results.append(combined_statistics)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    # results_df.sort_values(by='index', inplace=True)
    # results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv(results_path, index=False)

    pass


# @profile
def benchmark_MAC(sample, results_path):
    """Proxy for MAC Benchmarking Purposes."""
    # Benchmark CSP
    benchmark_solver(sample, SudokuSolverMAC, results_path)
    pass


# @profile
def benchmark_LSGA(sample, results_path):
    """Proxy for LSGA Benchmarking Purposes."""
    # Benchmark LSGA
    benchmark_solver(sample, SudokuSolverLSGA, results_path)
    pass


def benchmark_solvers_random_sample(df_path=df_path, sample_size: int = 1, min_clues: int = 17, max_clues: int = 80):
    """Perform a benchmark of SudokuSolverMAC and SudokuSolverLSGA on a random sample.

    Args:
        df_path (_type_, optional): _description_. Defaults to df_path.
        sample_size (int, optional): _description_. Defaults to 1.
        min_clues (int, optional): _description_. Defaults to 17.
        max_clues (int, optional): _description_. Defaults to 80.
    """
    # Output files suffix
    file_suffix = f'_rand_size_{sample_size}_min_{min_clues}_max_{max_clues}.csv'

    # Set output dir for LSGA
    lsga_results_path = os.path.join(
        resdir, 'lsga' + file_suffix
    )
    # Set output dir for CSP
    csp_results_path = os.path.join(
        resdir, 'csp' + file_suffix
    )

    # Load the dataset
    quiz_df = pd.read_csv(df_path)
    # Obtain random sample
    sample_df = get_random_df_sample(
        quiz_df, sample_size=sample_size,
        min_clues=min_clues, max_clues=max_clues
    )

    # Benchmark CSP
    benchmark_MAC(sample_df, csp_results_path)

    # Benchmark LSGA
    benchmark_LSGA(sample_df, lsga_results_path)

    pass


def join_csv_files_with_prefix(directory, prefix, output_path):
    """Join multiple CSV files.

    Args:
        directory (str): Directory containing the CSV files.
        prefix (str): Prefix string that filenames must start with.
        output_path (str): Path to save the resulting joined CSV file.
    """
    # Get list of CSV files in the directory with the given prefix
    file_paths = glob.glob(os.path.join(directory, f'{prefix}*.csv'))

    # List to hold each DataFrame
    dataframes = []

    # Load each matching CSV file into a DataFrame and append it to the list
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.sort_values(by='index', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_path, index=False)

    pass


if __name__ == '__main__':
    # # +++ Samples by Difficulty +++
    # Super-Easy
    benchmark_solvers_random_sample(sample_size=50, min_clues=61, max_clues=80)
    # Easy
    benchmark_solvers_random_sample(sample_size=50, min_clues=41, max_clues=60)
    # Medium
    benchmark_solvers_random_sample(sample_size=50, min_clues=26, max_clues=40)
    # Hard
    benchmark_solvers_random_sample(sample_size=50, min_clues=21, max_clues=25)
    # Expert
    benchmark_solvers_random_sample(sample_size=50, min_clues=17, max_clues=20)

    # # Join result datasets
    # MAC
    join_csv_files_with_prefix(resdir, 'csp_rand_', os.path.join(resdir, 'csp_benchmark.csv'))
    # LSGA
    join_csv_files_with_prefix(resdir, 'lsga_rand_', os.path.join(resdir, 'lsga_benchmark.csv'))

    pass
