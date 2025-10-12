# ---- Files ----
DEV_ENV := .env.dev
DEV_YML := docker-compose.dev.yml

PROD_ENV := .env.prod
PROD_YML := docker-compose.prod.yml

# ---- Helpers ----
define compose
	docker compose --env-file $(1) -f $(2)
endef

# ---- DEV ----
up-dev:
	$(call compose,$(DEV_ENV),$(DEV_YML)) up -d

down-dev:
	$(call compose,$(DEV_ENV),$(DEV_YML)) down -v

logs-dev:
	$(call compose,$(DEV_ENV),$(DEV_YML)) logs -f

ps-dev:
	$(call compose,$(DEV_ENV),$(DEV_YML)) ps

rebuild-dev:
	docker build -f airflow-docker/Dockerfile -t my-airflow:2.9.2 .
	docker build -f airflow-docker/Dockerfile.mlflow -t my-mlflow:2.14.1 .
	$(call compose,$(DEV_ENV),$(DEV_YML)) up -d --force-recreate

# ---- PROD ----
up-prod:
	$(call compose,$(PROD_ENV),$(PROD_YML)) up -d

down-prod:
	$(call compose,$(PROD_ENV),$(PROD_YML)) down -v

logs-prod:
	$(call compose,$(PROD_ENV),$(PROD_YML)) logs -f

ps-prod:
	$(call compose,$(PROD_ENV),$(PROD_YML)) ps

rebuild-prod:
	docker build -f airflow-docker/Dockerfile -t my-airflow:2.9.2 .
	docker build -f airflow-docker/Dockerfile.mlflow -t my-mlflow:2.14.1 .
	$(call compose,$(PROD_ENV),$(PROD_YML)) up -d --force-recreate