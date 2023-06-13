from collections import deque

import cv2
import mediapipe as mp
import numpy as np

RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
THICKNESS = 2
FONT_SIZE = 1


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


def main():
    # FPS счетчик
    cv_fps_calc = CvFpsCalc(buffer_len=5)

    # Координаты будущих точек
    coords = []

    mp_hands = mp.solutions.hands
    mp_myhand = mp_hands.Hands(max_num_hands=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)

    # Захват камеры
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Проверьте наличие камеры, или смените источник.")
        exit()

    while webcam.isOpened():

        _, frame = webcam.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        results = mp_myhand.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Берем только указательный палец
                finger_landmarks = hand_landmarks.landmark[8]
                finger_x = finger_landmarks.x
                finger_y = finger_landmarks.y
        else:
            finger_x, finger_y = None, None

        if finger_x:
            finger = (int(finger_x * image_width),
                      int(finger_y * image_height))

            cv2.circle(frame, finger, 2, RED_COLOR, THICKNESS)

            # Добавляем все координаты в список для последующей отрисовки
            coords.append(finger)

            # Рисуем линию по координатам пальца
            if len(coords) == 1:
                start_pt = finger
                end_pt = finger
                cv2.line(frame, start_pt, end_pt, RED_COLOR, THICKNESS)
            else:
                for pt in range(len(coords)):
                    if not pt + 1 == len(coords):
                        start_pt = (coords[pt][0], coords[pt][1])
                        end_pt = (coords[pt+1][0], coords[pt+1][1])
                        cv2.line(frame, start_pt, end_pt, RED_COLOR, THICKNESS)

        fps = cv_fps_calc.get()
        cv2.putText(frame, 'FPS=' + str(fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, BLUE_COLOR, THICKNESS)
        cv2.imshow('Board', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # закрытие камеры
    webcam.release()
    cv2.destroyAllWindows()

    # Рисуем итоговую картинку по всем точкам
    image = np.zeros((image_height, image_width, 3), np.uint8)
    for pt in range(len(coords)):
        if not pt + 1 == len(coords):
            cv2.line(image,
                     (coords[pt][0], coords[pt][1]),
                     (coords[pt + 1][0], coords[pt + 1][1]),
                     RED_COLOR, THICKNESS)

    # Показываем и сохраняем итоговую картинку
    cv2.imshow("krasnoe_na_4ernom", image)
    cv2.waitKey(0)
    cv2.imwrite("./result/krasnoe_na_4ernom.jpg", image)

    # Закрываем все окна
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
