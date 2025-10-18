# CI/CD

## Workflows

### CI (`ci.yml`)

Запускается при каждом пуше и PR на ветки `main`, `master`, `dev`.

**Шаги:**
1. Установка Python 3.12
2. Установка зависимостей через `uv`
3. Запуск тестов с покрытием
4. Загрузка отчёта покрытия в Codecov
5. Проверка кода через `ruff`

### Docker Build (`docker.yml`)

Запускается при пуше в `main`/`master` и при создании тега.

**Требования:**
- `DOCKER_USERNAME` в GitHub Secrets
- `DOCKER_PASSWORD` в GitHub Secrets

## Локальный запуск

```bash
uv run pytest tests/ -v --cov=src
uv run ruff check src/
```

