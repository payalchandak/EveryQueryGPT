#!/usr/bin/env bash
# shellcheck disable=SC2002,SC2086,SC2046
export $(cat .env | xargs)

WANDB_DIR=$PROJECT_DIR/models/.wandb

PYTHONPATH="$EVENT_STREAM_PATH:$PYTHONPATH" wandb agent $ENTITY/$PROJECT_NAME/$1 --count 1 