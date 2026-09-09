#!/usr/bin/env bash
# 从本报告保存的快照重建环境；保留原项目 .venv。
set -euo pipefail
evidence_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rebuild_dir="$(mktemp -d /tmp/llm-m0-rebuild-XXXXXX)"
printf '%s\n' "$rebuild_dir" > "$evidence_dir/rebuild_path.txt"
exec > >(tee "$evidence_dir/rebuild_log.txt") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$evidence_dir/rebuild_exit_code.txt"' EXIT
set -x
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
date -Is
uv --version
mkdir -p "$rebuild_dir/scripts"
cp "$evidence_dir/pyproject.toml" "$evidence_dir/uv.lock" \
   "$evidence_dir/.python-version" "$rebuild_dir/"
cp "$evidence_dir/check_training.py" "$rebuild_dir/scripts/"
uv sync --locked --directory "$rebuild_dir"
uv run --locked --directory "$rebuild_dir" python -c \
  'import sys, torch, numpy; print("Executable:", sys.executable); print("Prefix:", sys.prefix); print("Python:", sys.version); print("Torch:", torch.__version__); print("NumPy:", numpy.__version__); print("CUDA:", torch.version.cuda); assert sys.prefix == sys.argv[1] + "/.venv"; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))' "$rebuild_dir"
uv run --locked --directory "$rebuild_dir" python scripts/check_training.py \
  > "$evidence_dir/rebuild_output.txt" 2>&1
cat "$evidence_dir/rebuild_output.txt"
diff -u "$evidence_dir/output.txt" "$evidence_dir/rebuild_output.txt"
cmp "$evidence_dir/uv.lock" "$rebuild_dir/uv.lock"
printf '%s\n' 'PASS: fresh environment, GPU check, matching output, unchanged lockfile.'
