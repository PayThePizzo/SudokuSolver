import numpy as np
from collections import deque
from utils import print_sudoku


class SudokuSolverMAC:
    """Sudoku Solver using Maintaining Arc Consistency (MAC) and Backtracking.

    This class implements a solver for 9x9 Sudoku puzzles using constraint
    satisfaction with maintaining arc consistency (MAC) and backtracking.
    The solver reduces the possible values for each cell by propagating
    constraints across rows, columns, and 3x3 subgrids, and uses heuristics
    such as Most Constrained Variable (MRV) and Least Constraining Value (LCV)
    to guide the search for the solution.

    Attributes:
        empty_cell (int): Representation of an empty cell in the Sudoku grid,
            default is 0.
        full_domain (set): The set of all possible values for a Sudoku cell,
            {1, 2, ..., 9}.
        grid (list): The initial 9x9 Sudoku grid with 0 representing empty
            cells.
        domains (list): A 9x9 list of sets representing the possible values for
            each cell.
    """

    empty_cell = 0
    full_domain = set(range(1, 10))

    def __init__(self, grid):
        """Initialize the Sudoku solver with a given grid.

        Args:
            grid (list of lists): A 9x9 list of lists where each element is an
                integer from 0-9. Empty cells are represented by 0.
        """
        self.grid = grid
        self.domains = self.initialize_domains()

    def initialize_domains(self):
        """Initialize the domains for each cell in the 9x9 Sudoku grid.

        The domain of a cell represents the set of possible values (1 to 9)
        that the cell can take, based on the current grid. Initially, all cells
        have the full domain {1, 2, ..., 9}, except for cells that already
        have an assigned value.

        Returns:
            list of lists: A 9x9 list where each element is a set representing
                the possible values for the corresponding cell.
        """
        domains = [[set() for _ in range(9)] for _ in range(9)]
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == self.empty_cell:
                    domains[row][col] = self.full_domain.copy()
                else:
                    domains[row][col] = {self.grid[row][col]}
        return domains

    def preprocess_domains(self):
        """Preprocess the domains by performing initial constraint propagation.

        This method reduces the initial set of possible values for each cell by
        considering the values already assigned to neighboring cells in the
        same row, column, and subgrid. It applies the rules of Sudoku to narrow
        down possibilities.
        """
        for row in range(9):
            for col in range(9):
                if len(self.domains[row][col]) == 1:
                    self.update_domains(row, col, list(self.domains[row][col])[0])

    def update_domains(self, row, col, num):
        """Update the domains of other cells when a number is placed.

        This method updates the possible values (domains) of cells in the same
        row, column, and 3x3 subgrid as the cell where a number has been placed.

        Args:
            row (int): The row index of the cell where the number is placed.
            col (int): The column index of the cell where the number is placed.
            num (int): The number placed in the cell.
        """
        # Update row
        for c in range(9):
            if c != col:
                self.domains[row][c].discard(num)

        # Update column
        for r in range(9):
            if r != row:
                self.domains[r][col].discard(num)

        # Update 3x3 subgrid
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if r != row or c != col:
                    self.domains[r][c].discard(num)

    def get_neighbors(self, row, col):
        """Get all neighboring cells (same row, column, or 3x3 subgrid).

        This method returns the neighboring cells that share the same row,
        column, or 3x3 subgrid with the given cell. These neighbors are the
        ones affected by the value of the given cell.

        Args:
            row (int): Row index of the cell.
            col (int): Column index of the cell.

        Returns:
            set: A set of tuples representing the coordinates (row, col) of
                the neighboring cells.
        """
        neighbors = set()

        # Row and column neighbors
        for c in range(9):
            if c != col:
                neighbors.add((row, c))
        for r in range(9):
            if r != row:
                neighbors.add((r, col))

        # Box neighbors
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if (r, c) != (row, col):
                    neighbors.add((r, c))

        return neighbors

    def revise(self, xi, xj):
        """Revise the domain of cell xi to ensure arc consistency with cell xj.

        Arc consistency is ensured by checking if any value in xi's domain is
        inconsistent with xj's domain. If xj has only one possible value, that
        value is removed from xi's domain (if present).

        Args:
            xi (tuple): A tuple (row, col) representing the coordinates of the
                first cell.
            xj (tuple): A tuple (row, col) representing the coordinates of the
                second cell (neighbor).

        Returns:
            bool: True if xi's domain was revised (values were removed), False
                otherwise.
        """
        revised = False
        xi_row, xi_col = xi
        xj_row, xj_col = xj
        domain_xi = self.domains[xi_row][xi_col]
        domain_xj = self.domains[xj_row][xj_col]

        if len(domain_xj) == 1:
            single_value = next(iter(domain_xj))
            if single_value in domain_xi:
                domain_xi.discard(single_value)
                revised = True

        return revised

    def maintain_arc_consistency(self):
        """Ensure that the entire puzzle remains arc consistent.

        This method enforces arc consistency throughout the puzzle by using a
        queue of arcs (pairs of cells) and revising domains until no
        contradictions remain.

        Returns:
            bool: True if the domains are arc consistent, False if a
                contradiction is found (i.e., a cell has an empty domain).
        """
        queue = deque()
        # Initialize the queue with all arcs (each cell and its neighbors)
        for row in range(9):
            for col in range(9):
                neighbors = self.get_neighbors(row, col)
                for (nr, nc) in neighbors:
                    queue.append(((row, col), (nr, nc)))

        while queue:
            (xi, xj) = queue.popleft()
            if self.revise(xi, xj):
                if len(self.domains[xi[0]][xi[1]]) == 0:
                    return False  # Contradiction found

                # If xi's domain was revised, add its neighbors back to the queue
                for neighbor in self.get_neighbors(*xi):
                    if neighbor != xj:
                        queue.append((neighbor, xi))

        return True

    def find_most_constrained_cell(self):
        """Find the cell with the fewest remaining values (MRV heuristic).

        This method implements the Most Constrained Variable (MRV) heuristic
        to find the empty cell with the fewest remaining values in its domain,
        which is the cell most likely to cause a constraint violation.

        Returns:
            tuple: A tuple (row, col) representing the most constrained cell's
                coordinates, or (None, None) if no empty cells remain.
        """
        min_possibilities = 10  # More than any possible domain size
        chosen_cell = (None, None)

        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == self.empty_cell:
                    possible_values_count = len(self.domains[row][col])
                    if 0 < possible_values_count < min_possibilities:
                        min_possibilities = possible_values_count
                        chosen_cell = (row, col)

        return chosen_cell

    def get_least_constraining_values(self, row, col):
        """
        Get a sorted list of values for a cell based on the LCV heuristic.

        :param row: Row index of the cell.
        :param col: Column index of the cell.
        :return: A sorted list of integers (possible values) for the cell.
        """
        def count_restrictions(value):
            count = 0
            for neighbor in self.get_neighbors(row, col):
                if value in self.domains[neighbor[0]][neighbor[1]]:
                    count += 1
            return count

        return sorted(self.domains[row][col], key=count_restrictions)

    def solve(self):
        """
        Solve the Sudoku puzzle using MAC and backtracking.

        :return: True if the puzzle is solved, False otherwise.
        """
        # Apply initial constraint propagation to narrow down possible values
        self.preprocess_domains()

        row, col = self.find_most_constrained_cell()
        if row is None and col is None:
            return True

        possible_values = self.get_least_constraining_values(row, col)
        for value in possible_values:
            # Assign the value and make a copy of the current state
            self.grid[row][col] = value
            original_domains = [row.copy() for row in self.domains]

            # Update the domains and maintain arc consistency
            self.domains[row][col] = {value}
            if self.maintain_arc_consistency():
                if self.solve():
                    return True

            # Backtrack: revert the assignment and restore domains
            self.grid[row][col] = self.empty_cell
            self.domains = original_domains

        return False

    pass


if __name__ == "__main__":
    solver = SudokuSolverMAC(hard_problem_1)
    print('Puzzle: \n')
    print_sudoku(np.array(hard_problem_1))
    solver.solve()
    print('\n Solution: \n')
    print_sudoku(np.array(solver.grid))
    print('\n Proposed Solution: \n')
    print_sudoku(np.array(hard_solution_1))
    pass
