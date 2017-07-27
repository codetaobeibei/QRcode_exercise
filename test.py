from qrdetector import qrdetector
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
import logging; logging.basicConfig(level=logging.INFO)
import cv2

filepath = 'data/61d238c7jw1f35ewx12r0j20zk0qo41q.jpg'
img = cv2.imread(filepath)

def show(img, code=cv2.COLOR_BGR2RGB):
    cv_rgb = cv2.cvtColor(img, code)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(cv_rgb)
    plt.show()

Detector = qrdetector(img)

final_img = Detector.detector()

show(final_img)

