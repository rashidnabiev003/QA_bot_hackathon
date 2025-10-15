SHELL := /usr/bin/zsh

APP_IMAGE := qa-bot-app
BLEURT_IMAGE := bleurt-service:cpu

.PHONY: help build up down logs app-shell

help:
	@echo "Targets: build up down logs app-shell"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

app-shell:
	docker exec -it qa_bot_app /bin/sh

