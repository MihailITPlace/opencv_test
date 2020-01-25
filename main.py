import cv2
import new_algorithm as algo

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('Frame', frame)
    cv2.imshow('New Frame', algo.func(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()