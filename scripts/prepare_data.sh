#!/usr/bin/env bash
set -euo pipefail
echo "ascitwin reads a JSON manifest of per-patient records; see README Cohorts entry."
echo "target: ${1:-data/manifest.json}"
