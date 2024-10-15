from collections import deque


class SudokuSolver:
    empty_cell = 0
    full_domain = set(range(1, 10))

    def __init__(self, grid):
        """
        Initialize the Sudoku solver with a given grid.

        :param grid: A 9x9 list of lists with integers.
            Empty cells are represented by 0.
        """
        self.grid = grid
        self.domains = self.initialize_domains()
        pass

    def initialize_domains(self):
        """
        Initialize the domains for each cell in a 9x9 Sudoku grid.

        :return: A 9x9 list where each element is a set of possible values
            for the corresponding cell.
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
        """
        Preprocess the domains using constraint propagation to reduce initial
            possibilities.
        """
        for row in range(9):
            for col in range(9):
                if len(self.domains[row][col]) == 1:
                    self.update_domains(
                        row, col, list(self.domains[row][col])[0]
                    )
        pass

    def get_used_numbers(self, row, col):
        """
        Get a set of numbers that are already used in the given cell's row,
            column, and 3x3 box.

        :param row: The row index.
        :param col: The column index.
        :return: A set of integers that are already used.
        """
        used_numbers = set()

        # Check row and column
        used_numbers.update(self.grid[row])
        used_numbers.update(self.grid[r][col] for r in range(9))

        # Check 3x3 box
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                used_numbers.add(self.grid[r][c])

        # Remove empty cell representation
        used_numbers.discard(self.empty_cell)
        return used_numbers

    def update_domains(self, row, col, num):
        """
        Update the domains of other cells when a number is placed in a
            given cell.

        :param row: Row index of the placed number.
        :param col: Column index of the placed number.
        :param num: The number that was placed in the cell.
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

    def find_most_constrained_cell(self):
        """
        Find the empty cell with the fewest remaining values (MRV heuristic).

        :return: A tuple (row, col) of the most constrained cell's coordinates,
            or (None, None) if no empty cell.
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
            # Count how many times this value is present in
            # the domains of neighboring cells
            count = 0
            # Row and column
            for c in range(9):
                if c != col and value in self.domains[row][c]:
                    count += 1
            for r in range(9):
                if r != row and value in self.domains[r][col]:
                    count += 1
            # 3x3 box
            start_row, start_col = (row // 3) * 3, (col // 3) * 3
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    if (r != row or c != col) and value in self.domains[r][c]:
                        count += 1
            return count

        # Sort possible values based on the LCV heuristic
        return sorted(self.domains[row][col], key=count_restrictions)

    def solve(self):
        """
        Solve the Sudoku puzzle using constraint propagation and backtracking.

        :return: True if the puzzle is solved, False otherwise.
        """
        # Preprocess the domains before starting the backtracking search
        self.preprocess_domains()

        # Find the most constrained cell (MRV)
        row, col = self.find_most_constrained_cell()
        if row is None and col is None:
            return True  # Puzzle solved

        # Get possible values in LCV order
        possible_values = self.get_least_constraining_values(row, col)

        for value in possible_values:
            # Assign the value to the cell
            self.grid[row][col] = value
            # Save a copy of the domains before updating them
            original_domains = [row.copy() for row in self.domains]
            # Update the domains
            self.update_domains(row, col, value)

            # Recur to solve the rest of the puzzle
            if self.solve():
                return True

            # Backtrack: undo the assignment and restore domains
            self.grid[row][col] = self.empty_cell
            self.domains = original_domains

        return False

    def print_grid(self):
        """
        Print the current state of the Sudoku grid.
        """
        for row in self.grid:
            print(" ".join(str(num) if num != 0 else '.' for num in row))


class SudokuSolverMAC:
    empty_cell = 0
    full_domain = set(range(1, 10))

    def __init__(self, grid):
        """
        Initialize the Sudoku solver with a given grid.

        :param grid: A 9x9 list of lists with integers. Empty cells are
            represented by 0.
        """
        self.grid = grid
        self.domains = self.initialize_domains()

    def initialize_domains(self):
        """
        Initialize the domains for each cell in a 9x9 Sudoku grid.

        :return: A 9x9 list where each element is a set of possible values for
            the corresponding cell.
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
        """
        Preprocess the domains using initial constraint propagation to
            reduce initial possibilities.
        """
        for row in range(9):
            for col in range(9):
                if len(self.domains[row][col]) == 1:
                    self.update_domains(
                        row, col, list(self.domains[row][col])[0]
                    )

    def update_domains(self, row, col, num):
        """
        Update the domains of other cells when a number is placed in a
            given cell.

        :param row: Row index of the placed number.
        :param col: Column index of the placed number.
        :param num: The number that was placed in the cell.
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
        """
        Get all neighboring cells (in the same row, column, or 3x3 box).

        :param row: Row index.
        :param col: Column index.
        :return: A set of tuples (r, c) representing neighboring cell
            positions.
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
        """
        Revise the domain of xi to ensure arc consistency with xj.

        :param xi: Tuple (row, col) of the cell being revised.
        :param xj: Tuple (row, col) of the cell it is being compared against.
        :return: True if the domain of xi was revised, False otherwise.
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
        """
        Ensure that the entire puzzle remains arc consistent.

        :return: True if the domains are consistent, False if a
            contradiction is found.
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
                # If the domain of xi was revised to empty, there's a
                # contradiction
                if len(self.domains[xi[0]][xi[1]]) == 0:
                    return False

                # If xi's domain was revised, add all its neighbors back
                # to the queue
                for neighbor in self.get_neighbors(*xi):
                    if neighbor != xj:
                        queue.append((neighbor, xi))

        return True

    def find_most_constrained_cell(self):
        """
        Find the empty cell with the fewest remaining values (MRV heuristic).

        :return: A tuple (row, col) of the most constrained cell's coordinates,
            or (None, None) if no empty cell.
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

    def print_grid(self):
        """
        Print the current state of the Sudoku grid.
        """
        for row in self.grid:
            print(" ".join(str(num) if num != 0 else '.' for num in row))


class SudokuSolverCBJ:
    empty_cell = 0
    full_domain = set(range(1, 10))

    def __init__(self, grid):
        """
        Initialize the Sudoku solver with a given grid.

        :param grid: A 9x9 list of lists with integers. Empty cells are
            represented by 0.
        """
        self.grid = grid
        self.domains = self.initialize_domains()

    def initialize_domains(self):
        """
        Initialize the domains for each cell in a 9x9 Sudoku grid.

        :return: A 9x9 list where each element is a set of possible values for
            the corresponding cell.
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
        """
        Preprocess the domains using constraint propagation to
            reduce initial possibilities.
        """
        for row in range(9):
            for col in range(9):
                if len(self.domains[row][col]) == 1:
                    self.update_domains(
                        row, col, list(self.domains[row][col])[0]
                    )

    def get_used_numbers(self, row, col):
        """
        Get a set of numbers that are already used in the given cell's row,
            column, and 3x3 box.

        :param row: The row index.
        :param col: The column index.
        :return: A set of integers that are already used.
        """
        used_numbers = set()

        # Check row and column
        used_numbers.update(self.grid[row])
        used_numbers.update(self.grid[r][col] for r in range(9))

        # Check 3x3 box
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                used_numbers.add(self.grid[r][c])

        # Remove empty cell representation
        used_numbers.discard(self.empty_cell)
        return used_numbers

    def update_domains(self, row, col, num):
        """
        Update the domains of other cells when a number is placed in a
            given cell.

        :param row: Row index of the placed number.
        :param col: Column index of the placed number.
        :param num: The number that was placed in the cell.
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

    def find_most_constrained_cell(self):
        """
        Find the empty cell with the fewest remaining values (MRV heuristic).

        :return: A tuple (row, col) of the most constrained cell's coordinates,
            or (None, None) if no empty cell.
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
            # Count how many times this value is present in the
            # domains of neighboring cells
            count = 0
            # Row and column
            for c in range(9):
                if c != col and value in self.domains[row][c]:
                    count += 1
            for r in range(9):
                if r != row and value in self.domains[r][col]:
                    count += 1
            # 3x3 box
            start_row, start_col = (row // 3) * 3, (col // 3) * 3
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    if (r != row or c != col) and value in self.domains[r][c]:
                        count += 1
            return count

        # Sort possible values based on the LCV heuristic
        return sorted(self.domains[row][col], key=count_restrictions)

    def solve(self):
        """
        Solve the Sudoku puzzle using conflict-directed backjumping.

        :return: True if the puzzle is solved, False otherwise.
        """
        # Apply initial constraint propagation to narrow down possible values
        self.preprocess_domains()

        return self.backjump({}, [])

    def backjump(self, assignments, conflict_set):
        """
        Backjumping solver that uses the conflict set to guide the search.

        :param assignments: Dictionary mapping (row, col) to assigned value.
        :param conflict_set: List of conflicting cells.
        :return: True if solved, False if conflict occurs.
        """
        # Find the most constrained cell (MRV)
        row, col = self.find_most_constrained_cell()
        if row is None and col is None:
            return True  # Puzzle solved

        possible_values = self.get_least_constraining_values(row, col)

        for value in possible_values:
            if self.is_valid_assignment(row, col, value):
                # Assign value and update the grid and domains
                self.grid[row][col] = value
                assignments[(row, col)] = value
                original_domains = [row.copy() for row in self.domains]
                self.update_domains(row, col, value)

                # Try to solve the remaining grid
                if self.backjump(assignments, conflict_set):
                    return True

                # If conflict, revert and update conflict set
                self.grid[row][col] = self.empty_cell
                self.domains = original_domains

                # Add conflicting variables to the conflict set
                conflict_set.append((row, col))

        # If no valid assignment is found, backjump
        last_conflict = conflict_set.pop() if conflict_set else None
        if last_conflict:
            # Jump back to the conflicting cell
            row, col = last_conflict
            return False

        return False

    def is_valid_assignment(self, row, col, value):
        """
        Check if assigning the given value to the cell (row, col) is valid.

        :param row: The row index of the cell.
        :param col: The column index of the cell.
        :param value: The value to be assigned.
        :return: True if valid, False otherwise.
        """
        # Check row and column
        for c in range(9):
            if self.grid[row][c] == value:
                return False
        for r in range(9):
            if self.grid[r][col] == value:
                return False

        # Check 3x3 box
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if self.grid[r][c] == value:
                    return False

        return True

    def print_grid(self):
        """
        Print the current state of the Sudoku grid.
        """
        for row in self.grid:
            print(" ".join(str(num) if num != 0 else '.' for num in row))


if __name__ == "__main__":
    problem = None
    solution = None

    solver_1 = SudokuSolver(problem)
    solver_2 = SudokuSolverMAC(problem)
    solver_3 = SudokuSolverCBJ(problem)

    pass
