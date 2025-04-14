# Решение среды MountainCarContinuous с помощью PPO
![Обученная модель](/assets/video.gif)

## Для улучшения сходимости используются следующие приемы
* objective clipping - ограничивает слишком сильное обновление стратегии 
* Траектория для накопления опыта
* GAE для вычисления преимуществ
* Нормализация преимуществ
* Дисперсия - улучшает исследование среды

## Алгоритм 
Источник: John Schulman 2017 Proximal Policy Optimization Algorithms.
1. Собираем опыт в траекторию.
2. Отслеживание прогресса в тестовой среде.
3. Проверяем накопилась ли траектория.
  4. Вычисляем преимущества А, целевые значения для критика Q, старая лог-вероятность действий по старой политике π_old.
  5. Делим траекторию на батчи.
    6. Обновление критика с помощью Q.
    7. Отношение между новой и старой стратегиями ratio = π / π_old.
    8. Objective clipping.
    9. Обновляем актор с помощью A и ratio.
  10. Очистка траектории.

## Установка
pip install -r requirements.txt

## Запуск
Обучение
```bash
$ python train.py --help
usage: train.py [-h] [--cuda] -n NAME [-e ENV]

options:
  -h, --help            show this help message and exit
  --cuda                Enable CUDA
  -n NAME, --name NAME  Name of the run
  -e ENV, --env ENV     Environment id, default=MountainCarContinuous-v0
```

Играть
```bash
$ python play.py --help
usage: play.py [-h] -m MODEL

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model file to load
  ```

### Пример 1. Запуск тренировки
```bash
$ python train.py --cuda -n test1
```

### Пример 2. Запуск среды с тренированной моделью
```bash
$ python play.py --model best_90.dat
```

## Результаты
Одна видеокарта GeForce GTX 1650.
На обучение ушло 4 минуты.
Среднее вознаграждение 90 достигнуто за 240 тыс. шагов.

![Средняя награда за 100 шагов](/assets/result.png)