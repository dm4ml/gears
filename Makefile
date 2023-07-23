.PHONY: test release install

test:
	poetry run pytest

release:
	poetry version patch
	poetry publish --build

install:
	pip install poetry
	poetry install