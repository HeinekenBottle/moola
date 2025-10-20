import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class MoolaPaths(BaseModel):
    data: Path
    artifacts: Path
    logs: Path


def resolve_paths() -> MoolaPaths:
    data = Path(os.getenv("MOOLA_DATA_DIR", "./data")).resolve()
    artifacts = Path(os.getenv("MOOLA_ARTIFACTS_DIR", str(data / "artifacts"))).resolve()
    logs = Path(os.getenv("MOOLA_LOG_DIR", str(data / "logs"))).resolve()
    for p in (data, artifacts, logs):
        p.mkdir(parents=True, exist_ok=True)
    return MoolaPaths(data=data, artifacts=artifacts, logs=logs)
