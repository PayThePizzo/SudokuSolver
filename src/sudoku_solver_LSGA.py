"""Implementation Module of the Local Search Genetic Algorithm.

This implementation follows the pseudocode from the paper 'A
Novel Evolutionary Algorithm With Column and Sub-Block Local
Search for Sudoku Puzzles'.
"""

import sys
import os
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GAConfig

# Numpy Random Number Generator
rng = np.random.default_rng(12345)
random.seed(os.urandom(42))


class SudokuSolverLSGA(GAConfig):
    """Sudoku Solver LSGA Class using a Local Search Genetic Algorithm (LSGA).

    This class implements a Local Search Genetic Algorithm (LSGA) to solve
    Sudoku puzzles. The algorithm maintains a population of candidate
    solutions and employs evolutionary strategies such as selection,
    crossover, mutation, and local search to evolve the solutions over
    generations.

    Key Features:
        - **Population-Based Search**: Utilizes a population of Sudoku grids, allowing the
            algorithm to explore multiple potential solutions simultaneously.
        - **Initialization**: The population is initialized based on the provided
            Sudoku grid, preserving given numbers and ensuring unique values in each
            row.
        - **Fitness Evaluation**: Fitness is assessed based on the number of illegal
            columns and sub-blocks, encouraging valid Sudoku solutions.
        - **Tournament Selection**: Implements tournament selection to choose parent
            individuals for the next generation, promoting higher fitness solutions.
        - **Crossover and Mutation**: Combines genetic operators to create new
            individuals, ensuring that given numbers remain unchanged while
            introducing diversity.
        - **Local Search**: Includes column and sub-block local search strategies to
            resolve repeated numbers, improving solution quality.
        - **Elite Population Strategy**: Maintains an elite population to keep the best
            individuals and uses elite learning to enhance genetic diversity.
        - **Adaptability**: Configurable parameters allow for tuning the algorithm's
            behavior, including mutation rates and crossover probabilities.

    Attributes:
        log_dir (str): Path for storing log files related to the GA execution.
        solved (bool): A flag to track whether the algorithm successfully
            solved the puzzle.
        population_size (int): Total number of individuals in the population.
        elite_size (int): Number of individuals retained for the next generation.
        tournament_size (int): Number of individuals selected for tournament
            selection in each iteration.
        max_generations (int): Maximum number of generations allowed during
            the evolutionary process.
        individual_crossover_rate (float): Probability of crossover occurring
            for individual solutions (PC1).
        row_crossover_rate (float): Probability of crossover occurring for each
            row (PC2).
        swap_mutation_rate (float): Probability of mutation through cell
            swapping (PM1).
        reinitialization_mutation_rate (float): Probability of reinitializing
            rows during mutation (PM2).
        row_swaps (int): Count of row crossover swaps between parents.
        cell_swaps (int): Count of in-row cell swaps during mutation.
        row_reinitializations (int): Count of rows that have been
            reinitialized during mutation.
        col_swaps (int): Count of cell swaps between columns of the same row
            due to column local search.
        sub_swaps (int): Count of cell swaps between subblocks of the same row
            due to subblock local search.
        replacements (int): Count of the worst individuals replaced in the
            population.
        reinitializations (int): Count of the worst individuals reinitialized.
        generations (int): Current generation count during the evolutionary
            process.
        N (int): The size of the Sudoku grid (typically 9 for a standard Sudoku).
        grid (np.ndarray): A 9x9 numpy array representing the initial Sudoku grid.
        associated_matrix (np.ndarray): A binary matrix indicating which cells in the grid are pre-filled.
        logger (logging.Logger): Logger for tracking the execution and debugging.
    """

    def __init__(self, grid: list[list[int]]):
        """Initialize the Sudoku Solver.

        This function sets the common structure and values for
        the statistics used by the algorithm. These numbers
        are directly taken from the paper.

        Args:
            grid (list[list[int]]): An array-like 2D 9x9 list of integers
                representing the sudoku board puzzle.
        """
        # Grid length
        self.N = len(grid)
        # Input Sudoku Problem
        self.grid = np.array(grid)
        # Shared Associated Matrix
        self.associated_matrix = (self.grid > 0).astype(int)

        # Logger Setup
        self.logger = self.setup_logging(
            os.path.join(self.log_dir, 'LSGA_execution.log')
        )
        self.logger.info('Instance of SudokuSolverLSGA Initialized.')
        pass

    def initialize_row(self, matrix, row):
        """Initialize a row and return it."""
        # Original row
        new_row = np.copy(matrix[row, :])
        # Numbers to assign
        domain = np.setdiff1d(
            np.arange(1, 10),
            new_row[self.associated_matrix[row, :] == 1]
        ).tolist()
        # Shuffle numbers
        rng.shuffle(domain)

        # Assign the remaining numbers to the empty positions in the row
        for col in range(self.N):
            if self.associated_matrix[row, col] == 0:
                if len(domain) > 0:
                    # Assign first random element
                    new_row[col] = domain.pop()
                else:
                    raise IndexError(
                        'ERROR: Not enough unique numbers to assign.'
                    )

        if len(domain) > 0:
            raise Exception(
                'ERROR: domain is not empty after initialization.'
            )

        return new_row

    def create_individual(self):
        """Create a random individual based on the Sudoku grid.

        This method parses each row of the initial grid, it substitutes
        the 0 values with random numbers from 1 to 9 (except those which
        are assigned). This preserves the rule of rows in sudoku,
        namely 'All 1 to N numbers in each row should appear and not be
        repeated'.
        """
        # Major Matrix records the assigned numbers in each position
        major_matrix = np.copy(self.grid)

        for col in range(self.N):
            # Substitute row with initialized one
            major_matrix[col, :] = self.initialize_row(major_matrix, col)

        return major_matrix

    def initialize_population(self):
        """Initialize the population from the initial Sudoku grid.

        This method creates a list of tuples containing the major matrix
        and the associated matrix for each individual.
        """
        population = []

        # Create individuals
        for _ in range(self.population_size):
            major_matrix = self.create_individual()
            # Population is a tuple to keep track of
            # major matrix and fitness
            population.append((major_matrix, 0))

        return population

    def is_valid_component(self, component) -> bool:
        """Check if 1D array component is valid."""
        # Use 1D array
        component = np.copy(component).flatten()

        # Not valid if component has not the right length
        # Not valid if it contains 0
        # Not valid if numbers are missing or repeated
        if (
            len(component) != self.N or
            np.any(component == 0) or
            set(np.unique(component)) != set(range(1, 10))
        ):
            return False

        return True

    def is_valid_column(self, matrix, col: int) -> bool:
        """Check if the specified column is valid."""
        return self.is_valid_component(matrix[:, col])

    def is_valid_subblock(self, matrix, row, col) -> bool:
        """Check if the specified sub-block is valid."""
        return self.is_valid_component(
            matrix[row*3: row*3+3, col*3: col*3+3]
        )

    def fitness(self, matrix):
        """Calculate the fitness of the individual.

        This methods counts the number of illegal columns
        and illegal subblocks for a given individual. The sum of
        these two measures is the fitness score of the individual.
        It is not necessary to check the rows too as that condition
        is always respected.
        """
        fitness_score = 0

        # Evaluate columns (c_i)
        for col in range(self.N):
            if not self.is_valid_column(matrix, col):
                # Invalid Column
                fitness_score += 1

        # Evaluate sub-blocks (s_j)
        for row in range(3):
            for col in range(3):
                if not self.is_valid_subblock(
                    matrix, row, col
                ):
                    # Invalid Sub-block
                    fitness_score += 1

        return fitness_score

    def evaluate_population(self, population):
        """Calculate the fitness of the population."""
        evaluation = []

        # Get the evaluations from the fitness scores
        for major_matrix, _ in population:
            evaluation.append((major_matrix, self.fitness(major_matrix)))

        return evaluation

    def tournament_selection(self, population):
        """Perform tournament selection."""
        selected_individuals = []

        for _ in range(self.population_size):
            # Select two individuals randomly
            sample = rng.choice(a=range(len(population)), size=2)
            idx1 = sample[0]
            idx2 = sample[1]

            # Compare their fitness scores and select the better one
            if population[idx1][1] < population[idx2][1]:
                selected_individuals.append(population[idx1])
            else:
                selected_individuals.append(population[idx2])

        return selected_individuals

    def crossover(self, population):
        """Crossover ensuring fixed values remain unchanged."""
        new_population = []

        # Select parents based on individual crossover rate (PC1)
        for parent1, f1 in population:
            # Offspring 1
            offspring1 = np.copy(parent1)
            # Offspring 2
            offspring2 = None

            # rand1 < PC1
            if rng.random() < self.individual_crossover_rate:
                # Select second parent randomly
                parent2, f2 = random.choice(population)
                offspring2 = np.copy(parent2)

                # Select rows based on row crossover rate (PC2)
                for i in range(self.N):
                    # rand2 < PC2
                    if rng.random() < self.row_crossover_rate:
                        # Swap the selected row
                        offspring1[i, :], offspring2[i, :] = (
                            np.copy(offspring2[i, :]),
                            np.copy(offspring1[i, :])
                        )
                        # Update row swaps
                        self.row_swaps += 1
            # Save the offspring
            new_population.append((offspring1, f1))
            if offspring2 is not None:
                new_population.append((offspring2, f2))

        return new_population

    def mutate(self, population):
        """Mutation operation."""
        new_population = []

        # For each individual in the population
        for parent, fit in population:
            # Offspring
            offspring = np.copy(parent)

            # For each row in the individual
            for i in range(self.N):
                # Find indices for non-given numbers
                zero_indices = np.where(self.associated_matrix[i, :] == 0)[0]

                # rand1 < PM1 and the count of non-given numbers >= 2
                if (
                    len(zero_indices) >= 2 and
                    rng.random() < self.swap_mutation_rate
                ):
                    # Select two non-given numbers
                    cols = random.choices(
                        population=zero_indices,
                        k=2
                    )
                    j1 = cols[0]
                    j2 = cols[1]

                    del cols

                    # Swap position of values
                    offspring[i, j1], offspring[i, j2] = (
                        offspring[i, j2].copy(),
                        offspring[i, j1].copy()
                    )
                    # Update cell swaps
                    self.cell_swaps += 1

                # rand2 < PM2
                if rng.random() < self.reinitialization_mutation_rate:
                    # Substitue row with re-initialized one
                    offspring[i, :] = self.initialize_row(
                        np.copy(self.grid), i
                    )
                    # Update cell reinitializations
                    self.row_reinitializations += 1

            # Append the mutated individual
            new_population.append((offspring, fit))

        return new_population

    def find_repeated_indices(self, component):
        """Return the indices of repeated numbers in an array."""
        unique_values, counts = np.unique(component, return_counts=True)

        # Get the repeated values (those that appear more than once)
        repeated_values = unique_values[counts > 1]

        # Create a boolean mask for repeated values in the original component
        is_repeated = np.isin(component, repeated_values)

        # Get the indices of the repeated values in the original component
        repeated_indices = np.where(is_repeated)[0]

        return repeated_values, repeated_indices

    def column_local_search(self, population):
        """Perform column local search to eliminate repeated numbers."""
        new_population = []

        # For each individual
        for major_matrix, fit in population:
            offspring = np.copy(major_matrix)
            # Record Illegal Columns in set C
            illegal_columns_indices = [
                j for j in range(self.N)
                if not self.is_valid_column(offspring, j)
            ]

            # For each illegal column
            for j1 in illegal_columns_indices:
                # Randomly select another illegal column
                j2 = rng.choice(
                    illegal_columns_indices, 1, replace=False
                )[0]

                # Find repeated numbers and indices in col1 and col2
                repeated_values_col1, repeated_indices_col1 = (
                    self.find_repeated_indices(offspring[:, j1])
                )
                repeated_values_col2, repeated_indices_col2 = (
                    self.find_repeated_indices(offspring[:, j2])
                )

                # Find the common row indices for repeated numbers
                common_repeated_indices = np.intersect1d(
                    repeated_indices_col1, repeated_indices_col2
                )

                del repeated_indices_col1, repeated_indices_col2

                # Find the row indices for given numbers
                given_numbers_indices = np.union1d(
                    np.where(self.associated_matrix[:, j1] == 1)[0],
                    np.where(self.associated_matrix[:, j2] == 1)[0]
                )

                # Find the remaining indices
                candidate_indices = np.setdiff1d(
                    common_repeated_indices,
                    given_numbers_indices
                )

                del common_repeated_indices, given_numbers_indices

                # Try until no more candidates or first swap
                for i in candidate_indices:
                    # Check repeat numbers do not exist in both columns
                    if (
                        offspring[i, j1] not in repeated_values_col2 and
                        offspring[i, j2] not in repeated_values_col1
                    ):
                        # Swap cell values
                        offspring[i, j1], offspring[i, j2] = (
                            offspring[i, j2].copy(),
                            offspring[i, j1].copy()
                        )
                        # Update col swaps
                        self.col_swaps += 1
                        break

            new_population.append((offspring, fit))

        return new_population

    def subblock_local_search(self, population):
        """Perform subblock LS to eliminate repeated numbers."""
        new_population = []

        # For each individual
        for major_matrix, fit in population:
            offspring = np.copy(major_matrix)

            # Record Illegal Sub-block in set S
            illegal_subblocks_indices = []
            for start_row in range(3):
                for start_col in range(3):
                    if not self.is_valid_subblock(
                        offspring, start_row, start_col
                    ):
                        illegal_subblocks_indices.append(
                            (start_row, start_col)
                        )

            # For each illegal subblock
            for s1 in illegal_subblocks_indices:
                # Randomly select another illegal subblock
                s2 = rng.choice(
                    illegal_subblocks_indices, 1, replace=False
                )[0]

                # Unpack starting row and col
                s1_i, s1_j = s1
                s2_i, s2_j = s2

                # If starting rows do not match
                if s1_i != s2_i:
                    # Skip to avoid violations
                    continue

                # Flatten subblocks
                subblock1 = np.copy(
                    offspring[s1_i*3: s1_i*3+3, s1_j*3: s1_j*3+3]
                ).flatten()
                subblock2 = np.copy(
                    offspring[s2_i*3: s2_i*3+3, s2_j*3: s2_j*3+3]
                ).flatten()

                # Find repeated numbers and indices
                repeated_values_sub1, repeated_indices_sub1 = (
                    self.find_repeated_indices(subblock1)
                )
                repeated_values_sub2, repeated_indices_sub2 = (
                    self.find_repeated_indices(subblock2)
                )

                # Map subblock indices back to original matrix
                original_rows_sub1 = np.array([
                    s1_i*3 + (idx // 3)
                    for idx in repeated_indices_sub1
                ])
                original_rows_sub2 = np.array([
                    s2_i*3 + (idx // 3)
                    for idx in repeated_indices_sub2
                ])

                # Find common row indices
                common_rows = np.intersect1d(
                    original_rows_sub1, original_rows_sub2
                )

                # If no common row, skip to next iteration
                if len(common_rows) == 0:
                    continue

                original_cols_sub1 = np.array([
                    s1_j*3 + (idx % 3)
                    for idx in repeated_indices_sub1
                ])
                original_cols_sub2 = np.array([
                    s2_j*3 + (idx % 3)
                    for idx in repeated_indices_sub2
                ])

                del repeated_indices_sub1, repeated_indices_sub2

                # Fuse into a list of tuples and remove
                # tuples with row not in common row or given numbers
                original_indices_sub1 = [
                    (row, col)
                    for row, col in zip(original_rows_sub1, original_cols_sub1)
                    if (
                        row in common_rows and
                        self.associated_matrix[row, col] == 0
                    )
                ]
                original_indices_sub2 = [
                    (row, col)
                    for row, col in zip(original_rows_sub2, original_cols_sub2)
                    if (
                        row in common_rows and
                        self.associated_matrix[row, col] == 0
                    )
                ]

                swapped = False

                # Try until no more candidates or first swap
                for i1, j1 in original_indices_sub1:

                    # Break after first swap
                    if swapped is True:
                        break

                    for i2, j2 in original_indices_sub2:
                        # If the indices are for the same row
                        # and do not exist repeated in both subblocks
                        if (
                            i1 == i2 and
                            offspring[i1, j1] not in repeated_values_sub2 and
                            offspring[i2, j2] not in repeated_values_sub1
                        ):
                            # Swap cell values
                            offspring[i1, j1], offspring[i2, j2] = (
                                offspring[i2, j2].copy(),
                                offspring[i1, j1].copy()
                            )
                            # Swap made
                            swapped = True
                            # Update subblock swaps
                            self.sub_swaps += 1
                            # Exit
                            break

            new_population.append((offspring, fit))

        return new_population

    def update_elite_population(self, population, elite):
        """Update the elite population with the best individuals."""
        # Join the two groups
        new_elite = population + elite
        # Sort the individuals on fitness
        new_elite = sorted(
            new_elite,
            key=lambda x: x[1]
        )
        # Keep the 50 best individuals
        new_elite = new_elite[: 50]

        return new_elite

    def elite_population_learning(self, population, elite):
        """Replace the worst individual in the population."""
        # Sort the evaluated population by fitness in
        # descending order (higher is worse)
        new_population = sorted(
            population, key=lambda x: x[1], reverse=True
        )

        # Get the worst individual (highest fitness)
        x_worst, fitness_worst = new_population[0]

        # Select a random elite individual for potential replacement
        if len(elite) > 0:
            x_random, fitness_random = random.choice(
                elite
            )

            # Calculate Pb using the formula from the paper
            Pb = (
                (fitness_worst - fitness_random) / fitness_worst
                if fitness_worst > 0 else 0
            )

            # Replace the worst individual based on Pb or reinitialize
            if rng.random() < Pb:
                # Replace x_worst with x_random
                new_population[
                    new_population.index((x_worst, fitness_worst))
                ] = (x_random, fitness_random)
                # Update replacements
                self.replacements += 1
            else:
                # Reinitialize x_worst with a new random individual
                x_init = self.create_individual()
                fitness_init = self.fitness(x_init)
                new_population[
                    new_population.index((x_worst, fitness_worst))
                ] = ((x_init, fitness_init))
                # Update replacements
                self.reinitializations += 1

        return new_population

    def solve(self):
        """Return a solution for the sudoku puzzle."""
        # Track Generations
        count = 0
        # Track Best Individual and Its Fitness
        best_individual = None

        # Track solving
        self.solved = False

        # 1. First Initialization of Population
        self.logger.debug('Initialize Population for the first time.')
        self.population = self.initialize_population()

        # 2. First Evaluation of Population
        self.logger.debug('Evaluate Population for the first time.')
        self.population = self.evaluate_population(self.population)

        # Use a copy
        population = self.population.copy()

        # Initial Elite Individuals
        elite = []

        # 3. While block
        while count < self.max_generations:
            self.logger.debug(f'Starting Generation {count}')

            # 4. Tournament Selection
            self.logger.debug('Starting Tournament Selection')
            population = self.tournament_selection(population)
            self.logger.debug('End of Tournament Selection')

            # 5. Crossover
            self.logger.debug('Starting Crossover')
            population = self.crossover(population)
            self.logger.debug('End of Crossover')

            # 6. Mutation
            self.logger.debug('Starting Mutation')
            population = self.mutate(population)
            self.logger.debug('End of Mutation')

            # 7. Column Local Search
            self.logger.debug('Starting Column Local Search')
            population = self.column_local_search(population)
            self.logger.debug('End of Column Local Search')

            # 8. Sub-Block Local Search
            self.logger.debug('Starting Sub-block Local Search')
            population = self.subblock_local_search(population)
            self.logger.debug('End of Sub-block Local Search')

            # 9. Evaluate Population
            self.logger.debug('Starting Generation Evaluation')
            population = self.evaluate_population(population)
            self.logger.debug('End of Generation Evaluation')

            # Update Elite Individuals
            self.logger.debug('Starting Elite Population Update')
            elite = self.update_elite_population(population, elite)
            self.logger.debug('End of Elite Population Update')

            # 10. Elite Population Learning
            self.logger.debug('Starting Elite Population Learning')
            population = self.elite_population_learning(population, elite)
            self.logger.debug('End of Elite Population Learning')

            # 11. Reserve the best individual as g_best
            best_individual = min(elite, key=lambda x: x[1])
            self.logger.debug(f'Best Individual has Fitness {best_individual[1]}')

            # 12. Check if fitness of solution is optimal
            if best_individual[1] == 0:
                self.logger.info('Solution Found')
                self.logger.debug(f'Solution required {count} generations')
                self.solved = True
                break

            count += 1

        # Update generations used
        self.generations = count
        # Obtain best solution and its fitness
        return best_individual

    def get_statistics(self):
        """Return a dictionary of statistics.

        The statistics returned are:
            - `generations`, tracks the
                count of generations used;
            - `row_swaps`, counts how many
                times two parents swapped rows
            - `cell_swaps`, counts how many
                times an individual has a row
                mutation (swap of cell in a row)
            - `row_reinitializations`, counts how
                many times an individual' row is
                reinitialized during mutation
            - `col_swaps`, counts the swaps of values
                (in the same row) between columns
            - `sub_swaps`, counts the swaps of values
                (in the same row) between subblocks
            - `replacements`, tracks the replacements
                of the worst individuals with best ones
            - `reinitializations`, tracks the reinitializations
                of the worst individuals

        Returns:
            (dict): A dictionary with keys
                `generations`, `row_swaps`,
                `cell_swaps`,
                `row_reinitializations`,
                `col_swaps`, `sub_swaps`,
                `replacements`,
                `reinitializations`
        """
        stats_dict = {
            'generations': self.generations,
            'row_swaps': self.row_swaps,
            'cell_swaps': self.cell_swaps,
            'row_reinitializations': self.row_reinitializations,
            'col_swaps': self.col_swaps,
            'sub_swaps': self.sub_swaps,
            'replacements': self.replacements,
            'reinitializations': self.reinitializations
        }

        return stats_dict

    pass
