# ER-PLATFORM-001 · developer entry points
#
# Windows users without `make`: every target is a one-liner, and README.md
# lists the PowerShell equivalents.

.PHONY: help up down logs psql worker queue migrate downgrade revision rls-verify \
        test lint typecheck api reset seed-pincodes

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

up: ## Start Postgres (pgvector; also the task queue)
	docker compose up -d
	@echo "Waiting for health checks..."
	@docker compose ps

down: ## Stop the stack, keep the volumes
	docker compose down

reset: ## Stop the stack and DESTROY the data volumes
	docker compose down -v

logs: ## Tail both services
	docker compose logs -f

psql: ## psql as the OWNER role
	docker compose exec postgres psql -U er_owner -d entity_resolution

psql-app: ## psql as the DATA-PLANE role (RLS applies; expect errors without a bound tenant)
	docker compose exec postgres psql -U er_app -d entity_resolution

worker: ## Run the Postgres queue worker (all queues)
	python -m app.queue.worker --concurrency 4

queue: ## Show queue depth by queue and status
	docker compose exec postgres psql -U er_owner -d entity_resolution -c \
	  "SELECT queue, status, count(*) FROM task_queue GROUP BY 1, 2 ORDER BY 1, 2"

migrate: ## Apply all migrations (schema, then RLS)
	alembic upgrade head

downgrade: ## Roll back one migration
	alembic downgrade -1

revision: ## Autogenerate a migration: make revision m="add x"
	alembic revision --autogenerate -m "$(m)"

rls-verify: ## Re-run the RLS policy script and its guards against the live DB
	docker compose exec -T postgres psql -U er_owner -d entity_resolution \
	  -v ON_ERROR_STOP=1 -1 < migrations/sql/0002_rls_policies.sql

seed-pincodes: ## Bulk-load the All-India Pincode CSV: make seed-pincodes csv=path/to/file.csv
	python scripts/seed_pincodes.py --csv-path "$(csv)"

test: ## Integration tests — needs `make up && make migrate` first
	pytest -v

lint:
	ruff check app tests migrations scripts
	ruff format --check app tests scripts

typecheck:
	mypy app

api: ## Run the API locally
# --loop is REQUIRED on Windows and harmless elsewhere: uvicorn hardcodes
# ProactorEventLoop there and psycopg's async mode cannot use it, so every
# database-backed route 500s without this. See app/main.py.
	uvicorn app.main:app --reload --port 8000 --loop asyncio:SelectorEventLoop
