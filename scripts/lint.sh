set -e
set -x

ruff check src/handyllm tests
ruff format src/handyllm tests --check
