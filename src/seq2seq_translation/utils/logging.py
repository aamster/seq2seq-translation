import logging


def set_logging_level(logger, level):
    """Set logging level dynamically."""
    logger.setLevel(level)
    logger.setLevel(level)
    print(f"Logging level set to: {logging.getLevelName(level)}")
