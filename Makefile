run-tests:
	py.test tests/

init:
	pip install -r requirements.txt
	pip install -e .

dev: init
	pip install pytest
