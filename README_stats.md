# Скрипты визуализации статистики обучения

Два вспомогательных скрипта для построения графиков по логам, которые сохраняют
`pretrain.py` и `train.py`.

## `read_pretrain_stats.py`

Строит графики метрик **предобучения** (supervised) по эпохам.

- **Вход:** `pretrain_stats.csv` — файл, который сохраняет `pretrain.py` (колонка
  `stats` со списком словарей `{"ploss", "vloss", "accuracy", "value"}` на эпоху).
- Разбирает содержимое регуляркой (`re.findall`) и разворачивает вложенные списки
  в плоский `DataFrame` (по строке на шаг обучения).
- Убирает выбросы (`ploss > 4.5`, `vloss > 1`, `value < -0.2`) и линейно
  интерполирует пропуски.
- Строит 2×2 сетку графиков: Policy Loss, Value Loss, Accuracy, Value Prediction —
  тонкая линия «сырых» значений + скользящее среднее (MA(1000)).
- **Выход:** `training_metrics.png` + текстовая сводка (max accuracy, min losses
  по эпохам) в консоль.

### Запуск
```bash
python read_pretrain_stats.py
```
Файл `pretrain_stats.csv` должен лежать в той же папке (или поправьте путь в скрипте).

## `read_train_stats.py`

Строит графики метрик **self-play RL-обучения** (`train.py`).

- **Вход:** `stats_150000.csv` — чекпоинт статистики, который сохраняет `train.py`
  каждые `SAVE_EVERY` игр (колонки `games`, `time`, `endings`, `policy_loss`,
  `value_loss`, `entropy`).
- Строит 2×2 сетку графиков (Time, Policy Loss, Value Loss, Entropy) от количества
  сыгранных партий.
- Отдельно разбирает колонку `endings` (список счётчиков исходов: истечение ходов /
  победа белых / победа чёрных / ничья / пат) и строит их динамику на одном графике.
- **Выход:** `metrics.png` и `endings.png`.

### Запуск
```bash
python read_train_stats.py
```
По умолчанию читает `stats_150000.csv` — при использовании другого чекпоинта
поправьте имя файла в `pd.read_csv(...)`.

## Зависимости

Добавьте в `requirements.txt` проекта:
```
matplotlib>=3.7.0
```
(`pandas` и `numpy` уже указаны в основном `requirements.txt`.)
