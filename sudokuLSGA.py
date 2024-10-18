import os
import random
import numpy as np

# Numpy Random Number Generator
rng = np.random.default_rng(12345)


class SudokuLSGA:
    def __init__(
        self, grid,
        population_size: int = 150, elite_size: int = 50,
        max_generations: int = 200,
        individual_crossover_rate: float = 0.2,
        row_crossover_rate: float = 0.1,
        swap_mutation_rate: float = 0.3,
        reinitialization_mutation_rate: float = 0.05,
        tournament_size: int = 2
    ):
        self.N = len(grid)
        # Input Sudoku Problem
        self.grid = np.array(grid)
        # Shared Associated Matrix
        # Tracks if numbers were intentionally assigned
        self.associated_matrix = (self.grid > 0).astype(int)
        # Population Size
        self.population_size = population_size
        # Elite Poulation Size
        self.elite_size = elite_size
        # Tournament Size
        self.tournament_size = tournament_size
        # Maximum Generations Count
        self.max_generations = max_generations
        # PC1 or Individual Crossover Rate
        self.individual_crossover_rate = individual_crossover_rate
        # PC2 or Row Crossover Rate
        self.row_crossover_rate = row_crossover_rate
        # PM1 or Swap Mutation Rate
        self.swap_mutation_rate = swap_mutation_rate
        # PM2 or Reinitialization Mutation Rate
        self.reinitialization_mutation_rate = reinitialization_mutation_rate
        pass

    def create_individual(self):
        """Create a random individual based on the Sudoku grid.

        This method parses each row of the initial grid, it substitutes
        the 0 values with random numbers from 1 to 9 (except those who
        are given or assigned). This preserves the rule of rows in sudoku,
        namely 'All 1 to N numbers in each row should appear and not be
        repeated'.

        Returns:

        """
        # Major Matrix records the assigned numbers in each position
        major_matrix = np.copy(self.grid)

        for i in range(self.N):
            # Numbers to randomly assign
            row_domain = list(
                set(range(1, 10)) -
                set(
                    [
                        major_matrix[i, j]
                        for j in range(self.N)
                        if self.associated_matrix[i, j] == 1]
                )
            )

            # Suffle numbers for random assignment
            rng.shuffle(row_domain)

            # Assign the remaining numbers to the empty positions in the row
            for j in range(self.N):
                if self.associated_matrix[i, j] == 0:
                    major_matrix[i, j] = row_domain.pop()
        return major_matrix

    def initialize_population(self):
        """Initialize the population from the initial Sudoku grid.

        This method creates a list of tuples containing the major matrix
        and the associated matrix for each individual.

        Returns:

        """
        population = []

        # Create individuals (chromosomes)
        for _ in range(self.population_size):
            major_matrix = self.create_individual()
            # Population is a tuple to keep track of fitness
            population.append((major_matrix, 0))

        return population

    def is_valid_column(self, matrix, col):
        """Check if the specified column is valid."""
        numbers = set()
        for i in range(self.N):
            if matrix[i][col] != 0:
                if matrix[i][col] in numbers:
                    return False
                numbers.add(matrix[i][col])
        return True

    def is_valid_subblock(self, matrix, block_row, block_col):
        """Check if the specified sub-block is valid."""
        numbers = set()
        for i in range(block_row * 3, block_row * 3 + 3):
            for j in range(block_col * 3, block_col * 3 + 3):
                if matrix[i][j] != 0:
                    if matrix[i][j] in numbers:
                        return False
                    numbers.add(matrix[i][j])
        return True

    def fitness(self, major_matrix):
        """Calculate the fitness of the individual.

        This methods counts the number of illegal columns
        and illegal subblocks for a given individual. The sum of
        these two measures is the fitness score of the individual.
        It is not necessary to check the rows too as that condition
        is always respected.
        """
        fitness_score = 0

        # Evaluate columns (c_i)
        for j in range(self.N):
            if not self.is_valid_column(major_matrix, j):
                # Invalid Column
                fitness_score += 1

        # Evaluate sub-blocks (s_j)
        for block_row in range(3):
            for block_col in range(3):
                if not self.is_valid_subblock(
                    major_matrix, block_row, block_col
                ):
                    # Invalid Sub-block
                    fitness_score += 1

        return fitness_score

    def evaluate_population(self, population):
        """Calculate the fitness of the population."""
        evaluation = []
        # Get the evaluations from the fitness scores
        for i, major_matrix, association_matrix in enumerate(population):
            # Saves the index of the matrix in the population
            # and the fitness value
            evaluation.append((i, self.fitness(major_matrix)))
        return evaluation

    def tournament_selection(self, population, fitness_scores):
        """Perform tournament selection."""
        selected_individuals = []

        return selected_individuals

    def can_swap_rows(self, row1_maj, row1_ass, row2_maj, row2_ass):
        """Check if two rows can be swapped."""
        for j in range(self.N):
            # Ensure associated matrix rows are equal
            # Ensure major matrix have given number in the same position
            # TODO: Check condition
            if (
                (row1_ass[j] != row2_ass[j]) or
                (row1_ass[j] == 1 and row1_maj[j] != row2_maj[j])
            ):
                return False
        return True

    def crossover(self, population):
        """Crossover ensuring fixed values remain unchanged."""
        new_population = []
        random.seed(os.urandom(32))

        # Select parents based on individual crossover rate (PC1)
        for parent1_maj, parent1_ass in population:
            # Offspring 1
            offspring1_maj = np.copy(parent1_maj)
            offspring1_ass = np.copy(parent1_ass)
            # Offspring 2
            offspring2_maj = None
            offspring2_ass = None

            # rand1 < PC1
            if random.random() < self.individual_crossover_rate:
                # Select second parent randomly
                parent2_maj, parent2_ass = random.choice(population)
                offspring2_maj = np.copy(parent2_maj)
                offspring2_ass = np.copy(parent2_ass)

                # Select rows based on row crossover rate (PC2)
                for row in range(self.N):
                    # rand2 < PC2
                    if random.random() < self.row_crossover_rate:
                        # Check assigned numbers are in the same position
                        if self.can_swap_rows(
                            offspring1_maj[row], offspring1_ass[row],
                            offspring2_maj[row], offspring2_ass[row]
                        ):
                            # Swap the selected row
                            offspring1_maj[row], offspring2_maj[row] = (
                                offspring2_maj[row].copy(),
                                offspring1_maj[row].copy()
                            )

            # Save the offspring
            new_population.append((offspring1_maj, offspring1_ass))
            # FIXME: Modify how to save second offspring
            if offspring2_maj is not None:
                new_population.append((offspring2_maj, offspring2_ass))

        return new_population

    def mutate(self, population):
        """Mutation operation."""
        new_population = []

        # Improve randomness
        random.seed(os.urandom(16))

        # For each individual in the population
        for major_matrix, associated_matrix in population:

            # Offspring
            offspring_maj = np.copy(major_matrix)
            offspring_ass = np.copy(associated_matrix)

            # For each row in the individual
            for row in range(self.N):
                # Find indices for non-given numbers
                zero_indices = np.where(offspring_ass[row] == 0)[0]

                # rand1 < PM1
                if random.random() < self.swap_mutation_rate:
                    # If the count of non-given numbers >= 2
                    if len(zero_indices) >= 2:
                        # Select two non-given numbers
                        # TODO: Check they are different numbers?
                        # TODO: Check indices are in bound
                        cols = np.random.choice(zero_indices, 2, replace=False)
                        col1 = cols[0]
                        col2 = cols[1]
                        # Swap position of values
                        # TODO: Use copy() ?
                        offspring_maj[row][col1], offspring_maj[row][col2] = (
                            offspring_maj[row][col2],
                            offspring_maj[row][col1]
                        )

                # rand2 < PM2
                if random.random() < self.reinitialization_mutation_rate:
                    # Set of possible numbers
                    domain = set()
                    # Reinitialize the row for non-given numbers
                    for j in zero_indices:
                        # Add to set
                        domain.add(offspring_maj[row][j])
                        # Set back to 0
                        offspring_maj[row][j] = 0

                    # Shuffle domain
                    domain = sorted(
                        list(domain), key=lambda x: random.random()
                    )

                    # Randomly assign remaining numbers
                    # TODO: Check random assignment
                    for j in zero_indices:
                        offspring_maj[row][j] = domain.pop()

            # Append the mutated individual
            new_population.append((offspring_maj, offspring_ass))

        return new_population

    def can_swap_subblocks(self, matrix, block1, block2):
        """Check if two sub-blocks can be swapped."""
        block1_row, block1_col = block1
        block2_row, block2_col = block2

        # Check for repeat numbers in the same rows of the sub-blocks
        for i in range(block1_row * 3, block1_row * 3 + 3):
            if len(
                set(matrix[i, block1_col * 3:block1_col * 3 + 3]) &
                set(matrix[i, block2_col * 3:block2_col * 3 + 3])
            ) > 0:
                return False
        return True

    def column_local_search(self, population):
        """Perform column local search to eliminate repeated numbers."""
        new_population = []
        # Improve randomness
        random.seed(os.urandom(16))

        # For each individual
        for major_matrix, associated_matrix in population:
            offspring_maj = np.copy(major_matrix)
            offspring_ass = np.copy(associated_matrix)

            # Record Illegal Columns in set C
            illegal_columns_indices = [
                j for j in range(self.N)
                if not self.is_valid_column(offspring_maj, j)
            ]

            # For each illegal column
            for j1 in illegal_columns_indices:
                # Randomly select another illegal column
                j2 = np.random.choice(
                    illegal_columns_indices, 1, replace=False
                )

                # Get columns
                col1 = offspring_maj[:, j1]
                col2 = offspring_maj[:, j2]

                # Get occurrences
                occurrences_1 = {k: [] for k in range(1, 10)}
                occurrences_2 = {k: [] for k in range(1, 10)}

                for i, value in enumerate(col1):
                    occurrences_1[value].append(i)
                for i, value in enumerate(col2):
                    occurrences_2[value].append(i)

                # Get only repeated numbers' indices
                repetitions_1 = np.concatenate(
                    list(v for v in occurrences_1.values() if len(v) > 1),
                    axis=0
                ) if any(
                    len(v) > 1 for v in occurrences_1.values()
                ) else np.array([])

                repetitions_2 = np.concatenate(
                    list(v for v in occurrences_2.values() if len(v) > 1),
                    axis=0
                ) if any(
                    len(v) > 1 for v in occurrences_2.values()
                ) else np.array([])

                # Get indices with repetitions in common rows
                candidate_indices = (set(repetitions_1) & set(repetitions_2))

                # Try until no more candidates or first swap
                for i in list(candidate_indices):
                    # Ignore indices that belong to given numbers
                    if offspring_ass[i][j1] == 0 and offspring_ass[i][j2] == 0:
                        # Check repeat numbers do not exist in both columns
                        if (
                            len(occurrences_2[col1[i]]) <= 1 and
                            len(occurrences_1[col2[i]]) <= 1
                        ):
                            # Swap cell values
                            offspring_maj[i][j1], offspring_maj[i][j2] = (
                                offspring_maj[i][j2], offspring_maj[i][j1]
                            )
                            break
            new_population.append((offspring_maj, offspring_ass))
        return new_population

    def subblock_local_search(self, population):
        """Perform subblock local search to eliminate repeated numbers."""
        new_population = []
        # Improve randomness
        random.seed(os.urandom(16))

        # For each individual
        for major_matrix, associated_matrix in population:
            offspring_maj = np.copy(major_matrix)
            offspring_ass = np.copy(associated_matrix)

            # Record Illegal Sub-block in set S
            # TODO: sort or shuffle?
            illegal_subblocks_indices = []

            for block_row in range(3):
                for block_col in range(3):
                    if not self.is_valid_subblock(
                        offspring_maj, block_row, block_col
                    ):
                        illegal_subblocks_indices.append(
                            (block_row, block_col)
                        )

            # For each illegal subblock
            for s1 in illegal_subblocks_indices:
                # Randomly select another illegal subblock
                s2 = np.random.choice(
                    illegal_subblocks_indices, 1, replace=False
                )

                # Unpack starting row and col
                s1_i, s1_j = s1
                s2_i, s2_j = s2

                # If starting rows do not match
                if s1_i != s2_i:
                    # Skip to avoid violations
                    continue

                # Get occurrences
                occurrences_1 = {k: [] for k in range(1, 10)}
                occurrences_2 = {k: [] for k in range(1, 10)}

                for i in range((s1_i*3), (s1_i*3 + 3)):
                    for j in range((s1_j*3), (s1_j*3 + 3)):
                        key = offspring_maj[i][j]
                        occurrences_1[key].append((i, j))
                for i in range((s2_i*3), (s2_i*3 + 3)):
                    for j in range((s2_j*3), (s2_j*3 + 3)):
                        key = offspring_maj[i][j]
                        occurrences_2[key].append((i, j))

                # Get only repeated numbers' indices
                repetitions_1 = np.concatenate(
                    list(v for v in occurrences_1.values() if len(v) > 1),
                    axis=0
                ) if any(
                    len(v) > 1 for v in occurrences_1.values()
                ) else np.array([])

                repetitions_2 = np.concatenate(
                    list(v for v in occurrences_2.values() if len(v) > 1),
                    axis=0
                ) if any(
                    len(v) > 1 for v in occurrences_2.values()
                ) else np.array([])

                # Find common rows
                candidate_rows = (
                    {pos[0] for pos in repetitions_1} &
                    {pos[0] for pos in repetitions_2}
                )

                # Check swap made
                flag = False

                # Try until no more candidates or first swap
                for i1, j1 in repetitions_1:
                    # Break if swap has been made
                    if flag:
                        break

                    for i2, j2 in repetitions_2:
                        # If the indices are for the same row
                        # and are candidates
                        if i1 == i2 and i1 in candidate_rows:
                            # Ignore indices that belong to given numbers
                            if (
                                offspring_ass[i1][j1] == 0 and
                                offspring_ass[i2][j2] == 0
                            ):
                                # Check repeat numbers do not exist
                                # in both subblocks
                                if (
                                    len(
                                        occurrences_2[offspring_maj[i1][j1]]
                                    ) <= 1 and
                                    len(
                                        occurrences_1[offspring_maj[i2][j2]]
                                    ) <= 1
                                ):
                                    # Swap cell values
                                    (
                                        offspring_maj[i1][j1],
                                        offspring_maj[i2][j2]
                                    ) = (
                                        offspring_maj[i2][j2],
                                        offspring_maj[i1][j1]
                                    )
                                    # Swap made
                                    flag = True
                                    # Exit
                                    break
            new_population.append((offspring_maj, offspring_ass))
        return new_population

    def update_elite_population(self, evaluated_population):
        """Update the elite population with the best individuals."""

        pass

    def elite_population_learning(self, population):

        pass

    def evolve(self):
        """Main loop of the Genetic Algorithm."""
        # Track Generations
        count = 0
        # Track Best Individual and Its Fitness
        best_individual = None
        best_fitness = None

        # 1. First Initialization of Population
        self.population = self.initialize_population()
        # 2. First Evaluation of Population
        self.evaluation = self.evaluate_population(self.population)

        # Use a copy
        population = self.population.copy()
        evaluation = self.evaluation.copy()

        # Initial Elite Individuals
        # TODO: queue based on evaluation
        elite = None

        # 3. While block
        while count < self.max_generations:

            # 4. Tournament Selection
            # Get 100 individuals from population
            population = self.tournament_selection(
                population, evaluation
            )

            # Inject 50 elite individuals to population

            # 5. Crossover
            population = self.crossover(population)

            # 6. Mutation
            population = self.mutate(population)

            # 7. Column Local Search
            population = self.column_local_search(population)

            # 8. Sub-Block Local Search
            population = self.subblock_local_search(population)

            # 9. Evaluate Population
            evaluation = self.evaluate_population(population)

            # Update Elite Individuals
            elite = self.update_elite_population(self, population, evaluation)

            # 10. Elite Population Learning
            # Substitutes the worst individual

            # 11. Reserve the best individual as g_best
            # Sort new_population based on the evaluation
            best_individual = None
            best_fitness = self.fitness(best_individual)

            # 12. Check if fitness of solution is optimal
            if best_fitness == 0:
                break

            count += 1

        # Obtain best solution and its fitness
        return best_individual, best_fitness

    pass
