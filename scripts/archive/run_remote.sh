#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export MOOLA_ENV=remote
export MOOLA_DATA_DIR=${MOOLA_DATA_DIR:-"/workspace/data"}
export MOOLA_ARTIFACTS_DIR=${MOOLA_ARTIFACTS_DIR:-"/workspace/data/artifacts"}
export MOOLA_LOG_DIR=${MOOLA_LOG_DIR:-"/workspace/data/logs"}
moola doctor --over hardware=gpu
moola ingest --over hardware=gpu
moola train  --over hardware=gpu
moola evaluate --over hardware=gpu
moola deploy --over hardware=gpu
