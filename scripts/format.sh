set -x
set -e

ruff check src tests examples --fix
ruff format src tests examples
