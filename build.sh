#!/usr/bin/env bash
# Build the Vesta Docker image. Use this image in your own docker-compose elsewhere.
set -e
cd "$(dirname "$0")"
docker build -t vesta:latest -f app/Dockerfile .
