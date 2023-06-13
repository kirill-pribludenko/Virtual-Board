# Virtual-Board
 
Задача - реализовать виртуальную доску, на которой можно писать с помощью рук. Веб-камера должна отслеживать указательный палец и следить за его движением, оставляя след.

Приложение должно содержать следующий функционал
1. Считывание кадров с веб-камеры
2. Нахождение указательных пальцев, если они есть в кадре 
3. Рисовать на кадре траекторию движения пальца и давать таким образом писать 
4. Выводить обработанные кадры в отдельное окно


- 3 балла - обучена модель определения ключевых точек руки
- 4 балла - произведено сравнение с open source моделями
- 5 баллов - написано приложение

# Решение
В качестве датасета был выборан [Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)](http://domedb.perception.cs.cmu.edu/handdb.html)

Ноутбук с обучение и сравнение модели можно посмотреть тут [ссылка](./Hand_PE.ipynb)

В папке `result` можно найти пример сохраненной картинки, нарисованной на сделанной виртуальной доске.

Для **запуска** необходимо запустить `board.py`, предварительно установив все необходимые зависимости из `requirements.txt`

**Демонстрация** сервиса представлена ниже

<p align="left"><img src="./result/example.gif"\></p>
