setup:
	pip install pipenv
	pipenv run pip install pip==18.0
	pipenv install --dev --verbose

test:
	pipenv run python -m unittest discover