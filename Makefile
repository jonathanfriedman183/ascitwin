.PHONY: help install dev lint type test smoke docker clean

help:
	@grep -E "^[a-z]+:" Makefile | sed "s/:.*//"

install:
	pip install -e .

dev:
	pip install -e ".[dev,export]"

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	mypy

test:
	pytest

smoke:
	python -m ascitwin.console evaluate experiment=_smoke

docker:
	docker build -t ascitwin .

clean:
	rm -rf runs .mypy_cache .ruff_cache .pytest_cache *.egg-info
