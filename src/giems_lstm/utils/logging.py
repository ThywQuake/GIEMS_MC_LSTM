import logging
import sys


def _setup_global_logging(debug: bool, process_type: str):
    """
    Configures the root logger for the current process.
    - debug: If True, sets level to DEBUG. Otherwise, sets to INFO.
    - process_type: A label (e.g., 'Main' or 'Worker-ID') for log differentiation.
    """
    # set logging level based on debug flag
    level = logging.DEBUG if debug else logging.INFO

    # fetch the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid adding multiple handlers in multiprocessing
    if logger.handlers:
        print(f"Clearing {len(logger.handlers)} existing log handlers.")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Define format including process info for distinguishing logs in parallel runs
    formatter = logging.Formatter(
        f"[%(levelname)s] [%(asctime)s] [PROC:{process_type}] [%(filename)s:%(lineno)d] - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
