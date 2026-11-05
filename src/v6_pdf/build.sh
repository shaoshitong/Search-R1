#!/usr/bin/env bash
set -eu
document_source=$(cd "$(dirname "$0")" && pwd)
research_source=$(cd "$document_source/../.." && pwd)
output_pdf=${1:-"$research_source/docs/AI_CREATOR_GAME_RESEARCH_ZH.pdf"}
typst compile --root "$research_source" "$document_source/tex/report.typ" "$output_pdf"
