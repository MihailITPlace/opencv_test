import numpy as np
import cv2

cap = cv2.VideoCapture(0)


def canny_lsd(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(gray, 280, 360, apertureSize=3)

    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]

    if lines is None:
        return gray
    else:
        drawn_img = lsd.drawSegments(gray, lines)
        return drawn_img


#фильтер оттенков белого
# hsv_min = np.array((0, 0, 168), np.uint8)
# hsv_max = np.array((172, 111, 255), np.uint8)

#black
hsv_min = np.array([0, 0, 0])
hsv_max = np.array([255, 255, 75])

# Blue color
#hsv_min = np.array([94, 80, 2])
#hsv_max = np.array([126, 255, 255])

# Red color
#hsv_min = np.array([161, 155, 84])
#hsv_max = np.array([179, 255, 255])


def hsv_filter(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (11, 11), 0)
    hsv = cv2.inRange(hsv, hsv_min, hsv_max)
    cv2.imshow('hsv_filter', hsv)
    return hsv


def hsv_canny_lsd(frame):
    filtered_img = hsv_filter(frame)
    edges = cv2.Canny(filtered_img, 280, 360, apertureSize=3)

    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]

    if lines is None:
        return edges

    draw_2max_lines(frame, lines)
    drawn_img = lsd.drawSegments(edges, lines)
    return drawn_img


def length_of_line(line):
    a = line[0][0:2]
    b = line[0][2:4]
    return np.linalg.norm(a - b)


def tangents_of_line(line):
    if line is None:
        return 1

    line = line[0]
    x = line[0] - line[2]
    y = line[1] - line[3]

    if x == 0:
        return -1

    return y / x


def dist_between_lines(a_line, b_line):
    if (a_line is None) or (b_line is None):
        return MAX_DIST

    a_line = a_line[0]
    b_line = b_line[0]

    a1 = a_line[0:2]
    a2 = a_line[2:4]

    b1 = b_line[0:2]
    b2 = b_line[2:4]

    a = -1 * tangents_of_line([a_line])
    b = 1
    c = -1 * (a2[1] + a * a2[0])

    d1 = np.abs((a * b1[0] + b * b1[1] + c)) / (np.sqrt(a * a + b * b))
    d2 = np.abs((a * b2[0] + b * b2[1] + c)) / (np.sqrt(a * a + b * b))

    # d1 = np.linalg.norm(np.cross(a2-a1, a1-b1))/np.linalg.norm(a2-a1)
    # d2 = np.linalg.norm(np.cross(a2-a1, a1-b2))/np.linalg.norm(a2-a1)
    return min(d1, d2)


MIN_DIST = 15
MAX_DIST = 70
def find_2max_lines(lines):
    a_max_len = 0
    a_max_line = None

    b_max_len = 0
    b_max_line = None

    for line in lines:

        if np.abs(tangents_of_line(line)) < 1:
            continue

        new_len = length_of_line(line)

        if (new_len > a_max_len) and (MIN_DIST < dist_between_lines(line, b_max_line) <= MAX_DIST):
            #print('A:', dist_between_lines(line, b_max_line))
            a_max_len = new_len
            a_max_line = line
        elif (new_len > b_max_len) and (MIN_DIST < dist_between_lines(line, a_max_line) <= MAX_DIST):
            #print('B:', dist_between_lines(line, a_max_line))
            b_max_len = new_len
            b_max_line = line

    return a_max_line, b_max_line


def draw_2max_lines(frame, lines):
    a_line, b_line = find_2max_lines(lines)

    if a_line is None or b_line is None:
        return

    # a_tan = tan_line(a_line)
    # b_tan = tan_line(b_line)
    #
    # print('a line tan:', np.arctan(a_tan), a_tan)
    # print('b line tan:', np.arctan(b_tan), b_tan)

    draw_line(frame, a_line)
    draw_line(frame, b_line)


def draw_line(frame, line):
    if line is None:
        return
    line = line[0]
    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)


def hsv_canny_hough(frame):
    filtred_img = hsv_filter(frame)
    edges = cv2.Canny(filtred_img, 280, 360, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is None:
        return edges

    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)

    for line in lines:
        for r, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * r
            y0 = b * r

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))

            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return edges


def hsv_canny_houghP(frame):
    filtred_img = hsv_filter(frame)
    edges = cv2.Canny(filtred_img, 280, 360, apertureSize=3)

    minLineLength = 30
    maxLineGap = 50

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)

    if lines is None:
        return edges

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(edges, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return edges


while True:
    ret, frame = cap.read()

    new_frame = hsv_canny_lsd(frame)
    cv2.imshow('Frame', frame)
    cv2.imshow('New frame', new_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

