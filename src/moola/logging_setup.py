from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path, level: str = "INFO"):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level)
    logger.add(log_dir / "moola.log", level=level, rotation="10 MB", retention=3)
    return logger
