
import cv2
import glob

paths = glob.glob('data/hw/images/train/*.jpg')

fp = open('all_wh.txt', 'w', encoding='utf-8')

for path in paths:
    img = cv2.imread(path)
    h, w, _ = img.shape
    fp.write(f'{w}, {h}\n')

fp.close()