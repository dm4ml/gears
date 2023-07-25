.PHONY: test release install docs-release docs-serve

test:
	poetry run pytest

release:
	poetry version patch
	poetry publish --build

install:
	pip install poetry
	poetry install

docs-release:
	mkdocs gh-deploy --force

docs-serve:
	mkdocs serve