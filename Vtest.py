from qrdetector import qrdetector
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
import logging; logging.basicConfig(level=logging.INFO)
import cv2
import time

logging.info("camera warning up...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    Detector = qrdetector(frame)
    tri_box = Detector.detector(flag=1)
    if tri_box is not None:
        tri_box = np.int0(tri_box)
        frame = cv2.drawContours(frame, [tri_box], 0, (0, 255, 0), 2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
logging.info("exit")
