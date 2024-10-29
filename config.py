"""Sudoku Solver Configuration Module."""

import logging
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from os import path

basedir = path.abspath(path.dirname(__file__))
logdir = path.join(basedir, 'logs')
resdir = path.join(basedir, 'data', 'benchmark')
df_path = path.join(basedir, 'data', 'raw', 'sudoku_cluewise.csv')


class Config:
    """Base Configuration class for logging and performance tracking.

    This class sets up common logging, memory profiling, and performance
    tracking functions for applications, specifically for Sudoku solver
    subclasses.

    Attributes:
        logger (logging.Logger): A logger for recording application logs.
        solved (bool): A flag to track whether the algorithm successfully
            solved the puzzle.
    """

    solved = False

    @staticmethod
    def setup_logging(log_file_path, log_level=logging.INFO):
        """Set up logging configuration for the application.

        Initializes both file and console logging using a rotating file
        handler to limit file size.

        Args:
            log_file_path (str): The path where log files are stored.
            log_level (int): The logging level (e.g. logging.INFO)

        Log Message Format:
            - **%(asctime)s**: The date and time when the log message was
                created.
                - Format: `YYYY-MM-DD HH:MM:SS`
                - Example: `2024-08-02 14:55:23`
            - **%(name)s**: The name of the logger that generated the log
                message.
            - **%(levelname)s**: The severity level of the log message.
                - Examples: `INFO`, `DEBUG`, `WARNING`, `ERROR`, `CRITICAL`
            - **%(pathname)s**: The full pathname of the source file where the
                log message was generated.
                - Example: `/path/to/your/project/app.py`
            - **%(lineno)d**: The line number in the source file where the log
                message was generated.
                - Example: `42`
            - **%(message)s**: The log message content.
                - Example: `User login successful.`
            - **%(exc_info)s**: Exception information, if an exception is
                being logged.
                - Includes traceback and exception details.

        Returns:
            logging.Logger: Configured root logger instance.
        """
        # Format strings and Format handler
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = Formatter(fmt=fmt, datefmt=datefmt)

        # File handler for logging
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10240, backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Console handler for logging
        console_handler = StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)

        # Add file and console handlers if they aren't already added
        if not logger.hasHandlers():
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        # Log setup message
        logger.info(
            f'Logging initialized at {log_file_path} with ' +
            f'level {logging.getLevelName(log_level)}.'
        )

        return logger

    pass


class CSPConfig(Config):
    """Configuration class for the Constraint Propagation Sudoku Solver.

    This class serves as a base configuration for the Sudoku solver that
    utilizes constraint satisfaction techniques. It inherits from the
    BaseConfig class, enabling logging and performance tracking capabilities.
    The CSPConfig class defines parameters specific to constraint
    propagation, such as domains, empty cell representation, and
    statistics related to the solver's performance.

    Attributes:
        log_dir (str): Directory for storing log files related to the
            constraint satisfaction process.
        solved (bool): A flag to track whether the algorithm successfully
            solved the puzzle.
        empty_cell (int): The symbol used to represent an empty cell
            in the Sudoku grid (default is 0).
        full_domain (set): The set of valid values for a Sudoku cell,
            consisting of integers from 1 to 9.
        backtrack_steps (int): Counter for the number of backtracking
            steps taken during the solving process.
        forward_steps (int): Counter for the number of assignments made
            without requiring backtracking.
        ac3_revisions (int): Counter for the number of times the
            arc consistency algorithm revises the domains of variables.
        ac3_contradictions (int): Counter for the number of contradictions
            identified by the arc consistency algorithm.
        mrv_invocations (int): Counter for the number of times the
            Minimum Remaining Values (MRV) heuristic is invoked.
        recursive_calls (int): Counter for the number of recursive calls
            made during the solving process.
    """

    # Log directory
    log_dir = path.join(logdir, 'CSP')
    # Empty Cell Symbol
    empty_cell = 0
    # Full Domain
    full_domain = set(range(1, 10))
    # Number of backtracking steps
    backtrack_steps = 0
    # Number of forward steps (assignments without backtracking)
    forward_steps = 0
    # Number of times AC-3 revises an arc
    ac3_revisions = 0
    # Number of contradictions found by AC-3
    ac3_contradictions = 0
    # Times MRV heuristic is invoked
    mrv_invocations = 0
    # Recursive call count for the solve method
    recursive_calls = 0

    pass


class GAConfig(Config):
    """Configuration class for the Genetic Algorithm Sudoku Solver.

    This class inherits from the BaseConfig and provides configuration
    parameters specific to Genetic Algorithm (GA) solvers. It enables GA
    solvers to utilize logging and performance tracking functionalities
    while setting various parameters that govern the behavior of the
    genetic algorithm.

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
    """

    # Log directory
    log_dir = path.join(logdir, 'LSGA')
    # Population Size
    population_size = 150
    # Elite Poulation Size
    elite_size = 50
    # Tournament Size
    tournament_size = 2
    # Maximum Generations Count
    max_generations = 200
    # PC1 or Individual Crossover Rate
    individual_crossover_rate = 0.2
    # PC2 or Row Crossover Rate
    row_crossover_rate = 0.1
    # PM1 or Swap Mutation Rate
    swap_mutation_rate = 0.3
    # PM2 or Reinitialization Mutation Rate
    reinitialization_mutation_rate = 0.05

    # Crossover row swaps between parents
    row_swaps = 0
    # Mutation in-row cell swaps
    cell_swaps = 0
    # Mutation row reinitializations
    row_reinitializations = 0
    # Cell swaps between columns of same row
    # due to Column Local Search
    col_swaps = 0
    # Cell swaps between subblocks of same row
    # due to Subblock Local Search
    sub_swaps = 0
    # Count of worst indiduals replaced
    replacements = 0
    # Count of worst indiduals reinitialized
    reinitializations = 0
    # Generation count
    generations = 0

    pass
