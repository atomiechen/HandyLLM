set -x
set -e

ruff check src/handyllm tests --fix
ruff format src/handyllm tests
