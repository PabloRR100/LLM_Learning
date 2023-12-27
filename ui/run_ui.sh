#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

"${SCRIPT_DIR}/streamlit" run "${SCRIPT_DIR}/sandbox/ui/Home.py" \
  --server.headless=true  \
  --server.baseUrlPath=ui \
  --server.address=0.0.0.0
