#!/usr/bin/env bash
set -e
task_source_dir=$(cd "$(dirname "$0")/.." && pwd)
task_report_output=${1:-"$task_source_dir/docs/AI_CREATOR_GAME_REPORT.pdf"}
pandoc "$task_source_dir/docs/AI_CREATOR_GAME_REPORT.md" \
  --standalone --pdf-engine=typst --resource-path="$task_source_dir/docs" \
  -o "$task_report_output"
