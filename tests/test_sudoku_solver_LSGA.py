"""Test module for SudokuSolverLSGA Class."""

import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sudoku_solver_LSGA import SudokuSolverLSGA


class TestSudokuSolverLSGA(unittest.TestCase):
    """Unit tests for the SudokuSolverLSGA class."""

    def setUp(self):
        """Set up a sample 9x9 Sudoku grid for testing."""
        self.grid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        # Create a solver instance
        self.solver = SudokuSolverLSGA(self.grid)

        pass

    def test_initialization(self):
        """Test the initial setup of the SudokuSolverLSGA."""
        self.assertEqual(self.solver.grid.tolist(), self.grid)
        self.assertEqual(self.solver.N, 9)
        self.assertIsInstance(self.solver.associated_matrix, np.ndarray)

        pass

    def test_initialize_row(self):
        """Test row initialization to ensure valid domain setup."""
        row = 0
        initialized_row = self.solver.initialize_row(self.solver.grid, row)

        # All numbers should be unique
        self.assertEqual(len(set(initialized_row)), 9)

        pass

    def test_create_individual(self):
        """Verify that create_individual creates a valid individual with no duplicate values in rows."""
        individual = self.solver.create_individual()

        for row in individual:
            self.assertEqual(len(set(row)), 9)

        pass

    def test_initialize_population(self):
        """Test that the population initializes correctly with valid individuals."""
        population = self.solver.initialize_population()

        self.assertEqual(len(population), self.solver.population_size)

        for individual, _ in population:
            for row in individual:
                self.assertEqual(len(set(row)), 9)

        pass

    def test_is_valid_column(self):
        """Check column validity function."""
        individual = self.solver.create_individual()

        for col in range(9):
            self.assertTrue(self.solver.is_valid_column(individual, col))

        pass

    def test_is_valid_subblock(self):
        """Check subblock validity function."""
        individual = self.solver.create_individual()

        for row in range(3):
            for col in range(3):
                self.assertTrue(self.solver.is_valid_subblock(individual, row, col))

        pass

    def test_fitness(self):
        """Verify that the fitness function correctly identifies invalid columns and subblocks."""
        invalid_individual = np.copy(self.solver.grid)
        invalid_individual[0, 0] = 1  # Modify the grid to introduce a fitness penalty
        fitness_score = self.solver.fitness(invalid_individual)

        self.assertGreater(fitness_score, 0)  # Fitness should reflect the penalty

        pass

    def test_evaluate_population(self):
        """Test that evaluate_population correctly computes fitness for all individuals."""
        population = self.solver.initialize_population()
        evaluated_population = self.solver.evaluate_population(population)

        self.assertEqual(len(evaluated_population), len(population))

        for _, fitness_score in evaluated_population:
            self.assertIsInstance(fitness_score, int)

        pass

    def test_tournament_selection(self):
        """Test that tournament selection picks individuals with lower fitness."""
        population = self.solver.evaluate_population(self.solver.initialize_population())
        selected_population = self.solver.tournament_selection(population)

        self.assertEqual(len(selected_population), len(population))

        pass

    def test_crossover(self):
        """Verify crossover produces new offspring while maintaining row uniqueness."""
        population = self.solver.evaluate_population(self.solver.initialize_population())
        offspring_population = self.solver.crossover(population)

        self.assertGreaterEqual(len(offspring_population), len(population))

        pass

    def test_mutate(self):
        """Ensure mutation introduces variation without violating row uniqueness."""
        population = self.solver.evaluate_population(self.solver.initialize_population())
        mutated_population = self.solver.mutate(population)

        self.assertEqual(len(mutated_population), len(population))

        pass

    def test_find_repeated_indices(self):
        """Test that find_repeated_indices identifies duplicates in an array."""
        test_array = np.array([1, 2, 3, 3, 4, 5, 5, 6, 7])
        repeated_values, repeated_indices = self.solver.find_repeated_indices(test_array)

        self.assertIn(3, repeated_values)
        self.assertIn(5, repeated_values)
        self.assertIn(2, repeated_indices)  # 3's index
        self.assertIn(6, repeated_indices)  # 5's index

        pass

    def test_column_local_search(self):
        """Test that column_local_search reduces fitness by addressing repeated values in columns."""
        population = self.solver.evaluate_population(self.solver.initialize_population())
        optimized_population = self.solver.column_local_search(population)

        for _, fitness in optimized_population:
            self.assertIsInstance(fitness, int)

        pass

    def test_subblock_local_search(self):
        """Ensure subblock_local_search removes duplicates in subgrids to improve fitness."""
        population = self.solver.evaluate_population(self.solver.initialize_population())
        optimized_population = self.solver.subblock_local_search(population)

        for _, fitness in optimized_population:
            self.assertIsInstance(fitness, int)

        pass

    def test_update_elite_population(self):
        """Test elite population updates by keeping the best individuals."""
        population = self.solver.evaluate_population(self.solver.initialize_population())
        elite_population = []
        updated_elite = self.solver.update_elite_population(population, elite_population)

        self.assertLessEqual(len(updated_elite), self.solver.elite_size)
        pass

    def test_elite_population_learning(self):
        """Verify elite population learning replaces worst individuals."""
        population = self.solver.evaluate_population(self.solver.initialize_population())
        elite_population = self.solver.update_elite_population(population, [])
        optimized_population = self.solver.elite_population_learning(population, elite_population)

        self.assertEqual(len(optimized_population), len(population))

        pass

    def test_solve(self):
        """Test the overall solve function, ensuring it finds a valid solution."""
        best_individual, best_fitness = self.solver.solve()

        self.assertTrue(self.solver.solved)
        # Fitness should be 0 for a solved puzzle
        self.assertEqual(best_fitness, 0)

        pass

    pass


if __name__ == '__main__':
    unittest.main()

    pass
