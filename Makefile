init:
	pip install pipenv --upgrade
	pipenv install --dev
test:
	# This runs all of the tests
	pipenv install --dev pytest
	pytest
flake8:
	flake8 --count --select=E9,F63,F7,F82 --show-source --statistics cookeroo
	flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics cookeroo