import numpy as np
import cv2

# White
# hsv_min = np.array((0, 0, 168), np.uint8)
# hsv_max = np.array((172, 111, 255), np.uint8)

# Black
hsv_min = np.array([0, 0, 0])
hsv_max = np.array([255, 255, 75])

# Blue
#hsv_min = np.array([94, 80, 2])
#hsv_max = np.array([126, 255, 255])

# Red
#hsv_min = np.array([161, 155, 84])
#hsv_max = np.array([179, 255, 255])


def get_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
    hsv = cv2.inRange(hsv, hsv_min, hsv_max)

    hsv = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, np.ones([5, 5]))
    hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, np.ones([5, 5]))

    cv2.imshow('Mask', hsv)
    return hsv


def get_clahe(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(frame)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def func(frame):
    filtered_img = get_mask(get_clahe(frame))
    contours, hierarchy = cv2.findContours(filtered_img, 1, 2)

    h, w = filtered_img.shape
    rez = []

    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            filtered_img = cv2.circle(filtered_img, (cx, cy), 50, (0, 255, 0), 10)
            rez.append((cx, cy))

    turn = 0
    if len(rez) > 1:
        # y1 == y2
        if rez[0][1] - rez[1][1] == 0:
            a_r = 0
            b_r = w / 2
        # x1 == x2
        elif rez[1][0] - rez[0][0] == 0:
            a_r = 0
            b_r = rez[0][0]
        else:
            a = float(rez[0][1] - rez[1][1]) / float(rez[0][0] - rez[1][0])
            b = float(rez[0][1] - a * rez[0][0])
            filtered_img = cv2.line(filtered_img, (50, int(a * 50 + b)), (600, int(a * 600 + b)), (0, 255, 0), 2)
            a_r = 1 / a
            b_r = a_r * w - b / a
        turn += (b_r - w / 2) * 1
        turn += (- a_r) * 400
    if len(rez) == 1:
        turn += rez[0][0] - w / 2

    return filtered_img