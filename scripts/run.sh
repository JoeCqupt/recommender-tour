#!/usr/bin/env bash
# 一键编译并运行。用法: ./scripts/run.sh [topN]
set -euo pipefail
cd "$(dirname "$0")/.."

mvn -q compile
TOP_N="${1:-10}"
java -cp target/classes tour.recsys.Main data/ml-latest-small "$TOP_N"
