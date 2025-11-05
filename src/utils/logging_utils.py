"""
Logging utilities for Kolmogorov-Arnold networks experiments.

Provides standardized logging configuration with file and console handlers,
structured formatting, and configurable verbosity levels.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = 'KAN',
    log_file: Optional[Path] = None,
    level: str = 'INFO',
    quiet: bool = False,
    console_format: Optional[str] = None,
    file_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure structured logging for experimental pipeline.
    
    Creates a logger with:
        - File handler (if log_file specified)
        - Console handler (unless quiet=True)
        - Consistent formatting
        - Configurable verbosity
    
    Args:
        name: Logger name
        log_file: Path to log file (None to skip file logging)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        quiet: Suppress console output
        console_format: Custom format string for console
        file_format: Custom format string for file
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logging(
        ...     name='Experiment',
        ...     log_file=Path('output/logs/run.log'),
        ...     level='INFO',
        ...     quiet=False
        ... )
        >>> logger.info("Experiment started")
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Default format strings
    if console_format is None:
        console_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    
    if file_format is None:
        file_format = '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s'
    
    # Date format
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler (unless quiet)
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(
            logging.Formatter(console_format, datefmt=date_format)
        )
        logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(
            logging.Formatter(file_format, datefmt=date_format)
        )
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def setup_experiment_logging(
    output_dir: Path,
    experiment_name: str = 'experiment',
    level: str = 'INFO',
    quiet: bool = False
) -> logging.Logger:
    """
    Setup logging for a scientific experiment with automatic timestamping.
    
    Creates a timestamped log file in output_dir/logs/ and configures
    both file and console logging.
    
    Args:
        output_dir: Output directory for experiment
        experiment_name: Name prefix for log file
        level: Logging level
        quiet: Suppress console output
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_experiment_logging(
        ...     output_dir=Path('results/ablation'),
        ...     experiment_name='ablation',
        ...     level='INFO'
        ... )
    """
    output_dir = Path(output_dir)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_name}_{timestamp}.log'
    
    return setup_logging(
        name=experiment_name,
        log_file=log_file,
        level=level,
        quiet=quiet
    )


def get_logger(name: str = 'KAN') -> logging.Logger:
    """
    Get an existing logger or create a basic one if it doesn't exist.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up basic configuration
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(handler)
    
    return logger


def log_separator(logger: logging.Logger, char: str = '=', length: int = 80) -> None:
    """
    Log a separator line for visual organization.
    
    Args:
        logger: Logger instance
        char: Character to use for separator
        length: Length of separator line
    
    Example:
        >>> log_separator(logger, '=', 80)
        >>> logger.info("SECTION TITLE")
        >>> log_separator(logger, '=', 80)
    """
    logger.info(char * length)


def log_dict(
    logger: logging.Logger,
    data: dict,
    title: Optional[str] = None,
    level: str = 'INFO'
) -> None:
    """
    Log a dictionary with structured formatting.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        title: Optional title header
        level: Logging level for output
    
    Example:
        >>> config = {'n': 2, 'k': 6, 'gamma': 10.0}
        >>> log_dict(logger, config, title="Configuration")
    """
    log_func = getattr(logger, level.lower())
    
    if title:
        log_func(title)
    
    for key, value in data.items():
        if isinstance(value, dict):
            log_func(f"  {key}:")
            for k, v in value.items():
                log_func(f"    {k}: {v}")
        else:
            log_func(f"  {key}: {value}")


class LoggingContext:
    """
    Context manager for temporary logging configuration changes.
    
    Example:
        >>> logger = get_logger()
        >>> with LoggingContext(logger, level='DEBUG'):
        ...     logger.debug("This will be logged")
        >>> logger.debug("This won't be logged (back to original level)")
    """
    
    def __init__(self, logger: logging.Logger, level: Optional[str] = None):
        """
        Initialize context manager.
        
        Args:
            logger: Logger to modify
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper()) if level else None
        self.original_level = None
    
    def __enter__(self):
        """Enter context: save original level and set new level."""
        if self.new_level is not None:
            self.original_level = self.logger.level
            self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: restore original level."""
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)
        return False


def configure_external_loggers(level: str = 'WARNING') -> None:
    """
    Configure logging levels for external libraries to reduce noise.
    
    Sets common scientific libraries (matplotlib, PIL, etc.) to WARNING
    or higher to avoid cluttering experiment logs.
    
    Args:
        level: Logging level for external libraries
    
    Example:
        >>> configure_external_loggers('WARNING')
    """
    external_loggers = [
        'matplotlib',
        'matplotlib.font_manager',
        'PIL',
        'h5py',
        'numba',
        'jax',
        'torch'
    ]
    
    log_level = getattr(logging, level.upper())
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(log_level)


def main():
    """Test logging utilities."""
    import tempfile
    
    # Test basic setup
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / 'test.log'
        
        logger = setup_logging(
            name='Test',
            log_file=log_file,
            level='DEBUG',
            quiet=False
        )
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        log_separator(logger)
        
        test_config = {
            'model': {'n': 2, 'k': 6},
            'training': {'epochs': 100, 'lr': 1e-3}
        }
        log_dict(logger, test_config, title="Test Configuration")
        
        log_separator(logger)
        
        # Test context manager
        logger.setLevel(logging.INFO)
        logger.debug("This won't be logged")
        
        with LoggingContext(logger, level='DEBUG'):
            logger.debug("This will be logged (temporary DEBUG)")
        
        logger.debug("This won't be logged (back to INFO)")
        
        # Verify log file was created
        assert log_file.exists(), "Log file was not created"
        
        with open(log_file) as f:
            content = f.read()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Test Configuration" in content
        
        print(f"✓ All tests passed")
        print(f"  Log file: {log_file}")
        print(f"  Log size: {log_file.stat().st_size} bytes")


if __name__ == '__main__':
    main()