---

## Makefile

```makefile
.PHONY: setup up down fmt lint test

setup:
	poetry install
	pre-commit install

dvc:
	dvc repro

up:
	docker compose up -d --build

down:
	docker compose down

fmt:
	poetry run black .
	poetry run isort .

lint:
	poetry run ruff .
	poetry run mypy src

test:
	poetry run pytest -q
```

---