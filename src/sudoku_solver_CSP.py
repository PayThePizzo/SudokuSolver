import numpy as np
from collections import deque


class SudokuSolverMAC:
    """Sudoku Solver Class based on Constraint Propagation.

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
            list: A 9x9 list where each element is a set representing
                the possible values for the corresponding cell.
        """
        # 2D Matrix of empty sets for digits 1-9
        domain_matrix = []

        for row in range(9):
            # Row of the Matrix
            domain_row = []

            for col in range(9):
                # Domain of single variable
                domain_cell = None

                if self.grid[row][col] == self.empty_cell:
                    # Assign full domain to non-assigned cell
                    domain_cell = self.full_domain.copy()
                else:
                    # Limit domain of assigned cell
                    domain_cell = {self.grid[row][col]}
                
                if domain_cell is None:
                    raise ValueError(
                        "Domain cannot be None."
                    )

                domain_row.append(domain_cell)

            domain_matrix.append(domain_row)

        return domain_matrix

    def update_domains(self, row: int , col: int , num: int):
        """Update the domains of other cells when a number is placed.

        This method updates the possible values (domains) of cells in the same
        row, column, and 3x3 subgrid as the cell where a number has been placed.

        Args:
            row (int): The row index of the cell where the number is placed.
            col (int): The column index of the cell where the number is placed.
            num (int): The number placed in the cell.
        """
        # For each variable in same row or col
        for idx in range(9):
            if idx != col:
                # Restrict domains of same-row var
                self.domains[row][idx].discard(num)

            if idx != row:
                # Restrict domains of same-col var
                self.domains[idx][col].discard(num)

        # Find starting indices
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3

        # Update 3x3 subgrid
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if r != row or c != col:
                    # Restrict domain of same-grid var
                    self.domains[r][c].discard(num)
    
        pass 

    def preprocess_domains(self):
        """Preprocess the domains by performing initial constraint propagation.

        This Forward Checking method reduces the initial set of possible values
        for each cell by considering the values already assigned to neighboring
        cells in the same row, column, and subgrid.
        It applies the rules of Sudoku to narrow down possibilities.
        """
        # For each variable
        for row in range(9):
            for col in range(9):
                # If variable's domain is restricted to 1 choice
                if len(self.domains[row][col]) == 1:
                    # Restrict the domains of variables in same row, col and grid
                    self.update_domains(row, col, list(self.domains[row][col])[0])

        pass

    def get_neighbors(self, row: int, col: int):
        """Get all neighboring cells (same row, column, or 3x3 subgrid).

        This method returns the neighboring cells that share the same row,
        column, or 3x3 subgrid with the given cell. These neighbors are the
        ones affected by the value of the given cell.

        Args:
            row (int): Row index of the cell.
            col (int): Column index of the cell.

        Returns:
            set: A set of tuples representing
                the coordinates (row, col) of the neighboring cells.
        """
        neighbors = set()

        # Find row and column neighbors
        for idx in range(9):
            if idx != col:
                neighbors.add((row, idx))
        
            if idx != row:
                neighbors.add((idx, col))

        # Find starting indices
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3

        # Find 3x3 subgrid neighbors
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if (r, c) != (row, col):
                    neighbors.add((r, c))

        return neighbors

    def revise(self, xi: int, xj: int):
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

        # Find coordinated of Xi and Xj
        xi_row, xi_col = xi
        xj_row, xj_col = xj

        # Find domains of Xi and Xj
        domain_xi = self.domains[xi_row][xi_col]
        domain_xj = self.domains[xj_row][xj_col]

        # If no values of Xj allows to satisfy the constraint 
        if len(domain_xj) == 1:
            # Retrieve the single constrained value of Xj
            single_value = next(iter(domain_xj))
            # If present in the domain of Xi
            if single_value in domain_xi:
                # Remove it
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
        # Neighbor arcs queue
        queue = deque()

        # Initialize the queue with all arcs (each cell and its neighbors)
        for row in range(9):
            for col in range(9):
                # Find neighbors
                neighbors = self.get_neighbors(row, col)
                # For each neighbor add the arc
                for (nr, nc) in neighbors:
                    queue.append(((row, col), (nr, nc)))

        # Standard AC-3 Procedure
        while queue:
            (xi, xj) = queue.popleft()

            if self.revise(xi, xj):
                # Contradiction found
                if len(self.domains[xi[0]][xi[1]]) == 0:
                    return False

                for neighbor in self.get_neighbors(*xi):
                    # If xi's domain was revised
                    if neighbor != xj:
                        # add its neighbors back to the queue
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
        # Max possibilities
        min_possibilities = 10  
        # Cell to consider
        chosen_cell = (None, None)

        for row in range(9):
            for col in range(9):
                # Check if the cell is empty
                if self.grid[row][col] == self.empty_cell:
                    # Find its domain
                    possible_values_count = len(self.domains[row][col])

                    if 0 < possible_values_count < min_possibilities:
                        # Update possibilites and cell
                        min_possibilities = possible_values_count
                        chosen_cell = (row, col)

        return chosen_cell

    def get_least_constraining_values(self, row: int , col: int):
        """Get a sorted list of values for a cell based on the LCV heuristic.

        This method implements the Least Constraining Value (LCV) heuristic,
        which selects values that rule out the fewest possibilities for the
        neighboring cells. The returned list is sorted based on how
        constraining the values are.

        Args:
            row (int): Row index of the cell.
            col (int): Column index of the cell.

        Returns:
            list: A sorted list of integers (possible values) for the cell.
        """
        # Finds count for value
        def count_restrictions(value):
            count = 0
            # For each neighbor cell
            for neighbor in self.get_neighbors(row, col):
                # Update domain restriction count
                if value in self.domains[neighbor[0]][neighbor[1]]:
                    count += 1

            return count

        return sorted(self.domains[row][col], key=count_restrictions)

    def solve(self):
        """Solve the Sudoku puzzle using MAC and backtracking.

        This method applies constraint propagation and backtracking to solve
        the puzzle. It uses MAC to maintain arc consistency, and employs the
        MRV and LCV heuristics to guide the search.

        Returns:
            bool: True if the puzzle is solved, False if no solution exists.
        """
        # Apply initial constraint propagation to narrow down possible values
        self.preprocess_domains()

        # Find the variable with minimum remaining values 
        row, col = self.find_most_constrained_cell()

        # Puzzle solved if none found
        if row is None and col is None:
            return True 

        # Get the least constrainin
        possible_values = self.get_least_constraining_values(row, col)
        
        for value in possible_values:
            # Assign the value and make a copy of the current state
            self.grid[row][col] = value
            original_domains = [row.copy() for row in self.domains]

            # Update the domains and maintain arc consistency
            self.domains[row][col] = {value}

            # Maintain Arc Consistency Procedure
            if self.maintain_arc_consistency():
                # Keep going down the sol tree
                if self.solve():
                    return True

            # Backtrack: revert the assignment and restore domains
            self.grid[row][col] = self.empty_cell
            self.domains = original_domains

        return False


if __name__ == "__main__":
    solver = SudokuSolverMAC(hard_problem_3)
    print('Puzzle: \n')
    print_sudoku(np.array(hard_problem_3))
    solver.solve()
    print('\n Solution: \n')
    print_sudoku(np.array(solver.grid))
    print('\n Proposed Solution: \n')
    print_sudoku(np.array(hard_solution_3))
    pass
