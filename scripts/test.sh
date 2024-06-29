set -e
set -x

pytest --cov --cov-report=term-missing -o console_output_style=progress --cov-report=html
