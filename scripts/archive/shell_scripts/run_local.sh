#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export MOOLA_ENV=local
export MOOLA_DATA_DIR=${MOOLA_DATA_DIR:-"./data"}
export MOOLA_ARTIFACTS_DIR=${MOOLA_ARTIFACTS_DIR:-"./data/artifacts"}
export MOOLA_LOG_DIR=${MOOLA_LOG_DIR:-"./data/logs"}
moola doctor
moola ingest
moola train
moola evaluate
moola deploy
