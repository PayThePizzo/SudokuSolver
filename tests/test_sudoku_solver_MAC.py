"""Test module for SudokuSolverMAC Class."""

import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sudoku_solver_CSP import SudokuSolverMAC


class TestSudokuSolverMAC(unittest.TestCase):
    """Unit tests for the SudokuSolverMAC class."""

    def setUp(self):
        """Set up a sample 9x9 Sudoku grid for testing."""
        # Partially completed Sudoku puzzle
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
        self.solver = SudokuSolverMAC(self.grid)

        pass

    def test_initialization(self):
        """Test the initial setup of domains."""
        self.assertEqual(self.solver.grid, self.grid)
        self.assertIsInstance(self.solver.domains, list)
        self.assertEqual(len(self.solver.domains), 9)

        for row in self.solver.domains:
            self.assertEqual(len(row), 9)

        pass

    def test_initialize_domains(self):
        """Verify that domains are correctly initialized."""
        self.solver.initialize_domains()

        # Check specific cell domains to confirm initialization is accurate
        self.assertEqual(self.solver.domains[0][2], {1, 2, 4, 6, 8, 9})
        self.assertEqual(self.solver.domains[1][1], {2, 3, 4, 7, 8})
        self.assertEqual(self.solver.domains[8][0], {1, 2, 3, 4, 5, 6})

        pass

    def test_update_domains(self):
        """Verify domain updates when a cell value is set."""
        self.solver.update_domains(0, 0, 5)
        # Check cells in the same row, column, and 3x3 block
        self.assertNotIn(5, self.solver.domains[0][2])  # Same row
        self.assertNotIn(5, self.solver.domains[2][0])  # Same column
        self.assertNotIn(5, self.solver.domains[1][1])  # Same subgrid

        pass

    def test_preprocess_domains(self):
        """Test initial domain reduction via forward checking."""
        self.solver.preprocess_domains()
        # Check specific cells for correct domain reduction
        self.assertNotIn(5, self.solver.domains[0][2])  # Should be updated

        pass

    def test_get_neighbors(self):
        """Ensure correct neighbors are identified for a given cell."""
        neighbors = self.solver.get_neighbors(1, 1)
        self.assertIn((1, 0), neighbors)  # Row neighbor
        self.assertIn((0, 1), neighbors)  # Column neighbor
        self.assertIn((0, 0), neighbors)  # Subgrid neighbor
        self.assertNotIn((1, 1), neighbors)  # Should exclude itself

        pass

    def test_revise(self):
        """Check that revision is applied correctly to ensure arc consistency."""
        xi = (1, 1)
        xj = (0, 1)
        self.solver.domains[1][1] = {2, 3}
        self.solver.domains[0][1] = {3}
        revised = self.solver.revise(xi, xj)
        self.assertTrue(revised)
        self.assertNotIn(3, self.solver.domains[1][1])

        pass

    def test_maintain_arc_consistency(self):
        """Test that arc consistency is maintained across the puzzle."""
        self.solver.domains[1][1] = {2, 3}
        self.solver.domains[0][1] = {3}
        is_consistent = self.solver.maintain_arc_consistency()
        self.assertTrue(is_consistent)
        self.assertNotIn(3, self.solver.domains[1][1])

        pass

    def test_find_most_constrained_cell(self):
        """Check that the MRV heuristic correctly finds the most constrained cell."""
        row, col = self.solver.find_most_constrained_cell()
        self.assertIn((row, col), [(0, 2), (1, 1), (4, 4)])

        pass

    def test_get_least_constraining_values(self):
        """Test LCV heuristic for sorting values with minimal constraints."""
        values = self.solver.get_least_constraining_values(1, 1)
        self.assertIsInstance(values, list)
        self.assertEqual(set(values), self.solver.domains[1][1])

        pass

    def test_solve(self):
        """Check if the solver can solve a given puzzle correctly."""
        is_solved = self.solver.solve()
        self.assertTrue(is_solved)

        # Check for valid solution: All rows, cols, and 3x3 blocks should be unique
        for i in range(9):
            self.assertEqual(set(self.solver.grid[i]), set(range(1, 10)))  # Rows
            self.assertEqual(
                set(
                    self.solver.grid[row][i]
                    for row in range(9)
                ),
                set(range(1, 10))
            )  # Columns

        for idx_r in range(0, 9, 3):
            for idx_c in range(0, 9, 3):
                grid_np = np.array(self.solver.grid)
                block = grid_np[idx_r: idx_r+3][idx_c: idx_c+3].flatten()
                self.assertEqual(set(block), set(range(1, 10)))  # 3x3 subgrid

        pass


if __name__ == '__main__':
    unittest.main()

    pass
