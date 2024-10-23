"""Utils Module."""


def print_sudoku(board):
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
                    for num in board[i, j: j+3]
                ) for j in range(0, 9, 3)
            )
        )
        print(row)
