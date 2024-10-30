"""Utils module."""

import numpy as np


def string_to_sudoku_board(puzzle: str, empty_cell: str = '0', to_np_array: bool = False):
    """Convert a string into a 2D Sudoku board and return it.

    This function gathers the string and verifies its validity
    by checking:
        - The puzzle and the empty cell must be strings.
        - The empty
    if the empty
    cell symbol is coherent with the content of the string. Then
    it creates a 2D array of integers, substituting the empty cell
    symbol with a `0`.
    The returned array-like structure can be
    returned as `np.array`.

    Args:
        puzzle (str): A string of length 81, containing
            digits 1-9 and one empty cell symbol.
        empty_cell (str): A string representing the empty
            celll symbol.

    Raises:
        TypeError: When `puzzle` or `empty_cell` are not
            strings
        ValueError: When `puzzle` length is not exactly 81,
            or when the unique digits are not exactly 1-9 plus
            the empty cell symbol (hence 10).

    Returns:
        (list[list[int]] or np.array): An array-like 2D 9x9 list of integers
            representing the sudoku board puzzle.
    """
    # Check type
    if not isinstance(puzzle, str) or not isinstance(empty_cell, str):
        raise TypeError(
            'The arguments puzzle and empty_cell must be strings.'
        )

    # Check length
    if len(puzzle) != 81 or len(empty_cell) != 1:
        raise ValueError(
            'The argument puzzle must be exactly 81 digits long,\n' +
            'while empty_cell must be a string of ' +
            'exactly 1 digit.'
        )

    # Find the unique values in the string
    unique_values = set(puzzle)
    # Record the valid values for sudoku
    valid_values = set('123456789')
    valid_values.add(empty_cell)

    # Check if unique set is subset of valid set and length is respected
    if len(unique_values) > 10 or not (unique_values <= valid_values):
        raise ValueError(
            'The puzzle string must contain exactly 10 unique values:' +
            'digits 1-9 and one empty cell symbol. '
        )

    # Construct Board
    board = []

    for idx_r in range(9):
        row = []
        for idx_c in range(9):
            # Check value
            cell = puzzle[idx_r * 9 + idx_c]

            # Append correct value to row
            if cell == empty_cell:
                row.append(0)
            else:
                row.append(int(cell))

        board.append(row)

    # Convert to np.array
    if to_np_array:
        board = np.array(board)

    return board


def print_sudoku_board(puzzle) -> None:
    """Print the Sudoku board in a formatted way.

    Args:
        puzzle (np.ndarray): A 9x9 matrix representing the Sudoku board.

    Returns:
        None
    """
    for i in range(9):
        # Print horizontal lines for sub-grids
        if i % 3 == 0 and i != 0:
            print('-' * 21)
        # Print each row
        row = (
            ' | '.join(
                ' '.join(
                    str(num)
                    if num != 0 else '.'
                    for num in puzzle[i, j: j+3]
                ) for j in range(0, 9, 3)
            )
        )
        print(row)

    pass


def is_valid_solution(board) -> bool:
    """Check if a given 9x9 numpy array represents a valid Sudoku solution.

    Args:
        board (np.array): A 9x9 grid of integers.

    Returns:
        bool: True if the board is a valid Sudoku solution, False otherwise.
    """
    # Check if the shape is 9x9
    if board.shape != (9, 9):
        return False

    # Check each row, column, and 3x3 subgrid for validity
    for i in range(9):
        # Check if each row and column contain numbers 1 to 9 without repetition
        if not (np.array_equal(np.sort(board[i, :]), np.arange(1, 10)) and
                np.array_equal(np.sort(board[:, i]), np.arange(1, 10))):
            return False

        # Check the 3x3 subgrid
        row_offset = (i // 3) * 3
        col_offset = (i % 3) * 3
        subgrid = board[row_offset:row_offset+3, col_offset:col_offset+3].flatten()
        if not np.array_equal(np.sort(subgrid), np.arange(1, 10)):
            return False

    return True
