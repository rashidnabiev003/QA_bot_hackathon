# Инструкция по развертыванию

## Проект собран на базе следующей системы

- Linux(Arch/Manjaro)
- CPU: 20 ядер
- RAM: 32+ GB
- GPU: 2x5070ti (32 GB)(1 для vllm контейнера, 1 для всего остального) 

### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/your-repo/QA_bot_hackathon.git
cd QA_bot_hackathon
```

### Шаг 2: Подготовка окружения

Создайте файл `.env` в корне проекта:


### Шаг 3: Подготовка моделей

#### BLEURT-20

```bash
mkdir -p models
cd models
git clone https://huggingface.co/Elron/bleurt-base-128 BLEURT-20
cd ..
```

#### Ollama модель

Модель загружается после запуска контейнера(приколы конкретно моей системы, но если в логах будет кричать, что модели нет, то это точно поможет):

```bash
docker compose up -d ollama
docker exec -it qa_bot_ollama ollama pull gpt-oss:20b
```

### Шаг 4: Настройка документов

Поместите ваш документ в формате DOCX:

```bash
cp your_document.docx src/data/input.docx
```

Для бенчмарка подготовьте XLSX файл с колонками:
- id
- question
- answer
- context

Пример готового файла уже находится в данном каталоге :
```bash
cp your_triplets.xlsx src/data/base_triplets.xlsx
```
### Шаг 5: Запуск сервисов

#### Чтобы запустить сборку и запуск всех сервисов в корне проекта запустите

```bash
make up
```
### !!!Обязательно
Всегда запускайте проект с логами(в отдельном окне желательно, так как перед первым сообщением - придется ждать от 2 до 10), так как по началу,  очень долго будут скачиваться модели и собираться контейнеры.
Если запускаете весь проект, то дождитесь в логах полного развертывания vllm контейнера, а потом можно тыкать на сообщения/сборку индекса/ запуск бенчмарка.

```bash
make logs
```
#### vLLM

Ограничение видимости видеокарт(можете не использовать, если у вас одна больша карта ~20GB+, в моем случае пришлось делать такие приколы):
Запуск VLLM опциональное решение, для проверки гипотезы с llm reranking-ом, без нужды можно и не включать сервис, так как финальное решение было собрано без llm_reranking.

```bash
CUDA_VISIBLE_DEVICES=0     # Основное приложение
NVIDIA_VISIBLE_DEVICES=1   # vLLM
```

В `docker-compose.yml`:

```yaml
app:
  environment:
    - CUDA_VISIBLE_DEVICES=0

vllm:
  environment:
    - NVIDIA_VISIBLE_DEVICES=1
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```


Для работы на CPU установите:

```bash
SAS_DEVICE=cpu
SAS_FORCE_CPU=1
EMB_ON_CUDA=0
```

## Управление индексом

### Создание индекса

Индекс создается автоматически при первом запросе, если `AUTO_BUILD_INDEX=1`, но в example он стоит значением 0, поэтому удобно пересоздать в UI соответствующей кнопкой


### Обновление документов

1. Замените `src/data/input.docx`
2. Пересоберите индекс:
Или перезапустите контейнер с `FORCE_REBUILD_INDEX=1`.

## Бенчмарк

### Подготовка данных

Создайте XLSX файл с колонками:
- `id` - уникальный идентификатор
- `question` - вопрос
- `answer` - эталонный ответ (для генеративных метрик)
- `context` - эталонный контекст (для retrieval метрик)

Реальный пример есть в `src/data/base_triplets.xlsx`
### Запуск через UI

1. Откройте http://localhost:8000
2. Нажмите "Бенчмарк XLSX"( можно выбрать максимально количество колонок)
3. Результаты сохраняются в `src/data/benchmark_results.xlsx`

## Troubleshooting

### Ollama не отвечает

```bash
docker exec -it qa_bot_ollama ollama list
docker exec -it qa_bot_ollama ollama pull gpt-oss:20b
```

### vLLM out of memory

Уменьшите параметры в `docker-compose.yml`:

```yaml
--max-model-len 1024
--gpu-memory-utilization 0.6
```

### BLEURT сервис недоступен

Проверьте наличие модели:

```bash
ls -la models/BLEURT-20/
docker compose logs bleurt
```

### Индекс не создается

Проверьте наличие input.docx:

```bash
ls -la src/data/input.docx
docker compose logs app | grep -i index
```

Пересоберите вручную:

```bash
docker exec -it qa_bot_app sh
cd /app
python -c "from src.pipeline.orchestrator import get_pipeline; p = get_pipeline(); p.rebuild_index()"
```

### Ошибки импорта моделей

Проверьте подключение к Hugging Face:

```bash
export HF_ENDPOINT=https://huggingface.co
export HUGGING_FACE_HUB_TOKEN=your_token
```

Или используйте зеркало:

```bash
export HF_ENDPOINT=http://5.129.204.167:8081
```

## Мониторинг

### Логи

```bash
docker compose logs -f app
docker compose logs -f vllm
docker compose logs -f ollama
docker compose logs -f bleurt
```

### Метрики контейнеров

```bash
docker stats qa_bot_app qa_bot_vllm qa_bot_ollama bleurt20
```
