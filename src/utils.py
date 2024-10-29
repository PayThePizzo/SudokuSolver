"""Utils module."""

import numpy as np


def string_to_sudoku_board(puzzle: str, empty_cell: str = '0') -> list[list[int]]:
    """
    Convert a Sudoku string into a 2D 9x9 list of integers (board).

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
        list[list[int]]: An array-like 2D 9x9 list of integers
            representing the sudoku board puzzle.
    """
    # Check the type is correct
    if not isinstance(puzzle, str) or not isinstance(empty_cell, str):
        raise TypeError(
            'Both the puzzle and the empty cell must be strings.'
        )

    # Check if the string is exactly 81 characters (a 9x9 Sudoku board)
    if len(puzzle) != 81:
        raise ValueError(
            'The puzzle string must be exactly 81 characters long.'
        )

    # Find the unique values in the string
    unique_values = set(puzzle)
    # Record the valid values for sudoku
    valid_values = set('123456789')
    valid_values.add(empty_cell)

    # Check if unique set is subset of valid set and length is respected
    if len(list(unique_values)) > 10 or not (unique_values <= valid_values):
        raise ValueError(
            'The puzzle string must contain exactly 10 unique values:' +
            'digits 1-9 and one empty cell symbol. '
        )

    del unique_values, valid_values

    # Construct Board
    board = []
    step = 9
    for row in range(step):
        chunk = []
        for col in range(step):
            cell = puzzle[row * step + col]
            # Append value to row
            if cell == empty_cell:
                chunk.append(0)
            else:
                chunk.append(int(cell))

        board.append(chunk)

    return board


def print_sudoku_board(puzzle) -> None:
    """Print the Sudoku board in a formatted way.

    Args:
        board (np.ndarray): A 9x9 matrix representing the Sudoku board.
    """
    for i in range(9):
        # Print horizontal lines for sub-grids
        if i % 3 == 0 and i != 0:
            print('-' * 21)  # 21 dashes for 9 columns with spaces
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
    """
    Check if a given 9x9 numpy array represents a valid Sudoku solution.

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
