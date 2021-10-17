setup:
	poetry self update
	poetry install --no-interaction
	poetry run pip install -e .

test:
	poetry run python -m unittest discover