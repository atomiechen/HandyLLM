set -e
set -x

ruff check src tests examples
ruff format src tests examples --check --diff
