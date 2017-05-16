import numpy as np
import cv2
import image_recognition

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    
    borders = image_recognition.find_screen_borders(frame)
    cv2.drawContours(frame, [borders], 0, (255, 0, 0), 3) 

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
