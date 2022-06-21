import cv2
import glob
import json
from unittest import result
import numpy as np

label_files = glob.glob('data/hw/labels/val/*.txt')

result = {
    'annotations': [],
    'categories': [],
    'images': [],
}

for i in range(10):
    result['categories'].append(
        {
            'id': i,
            'name': str(i),
        }
    )

cnt = 0
fp = open('result.json', 'w', encoding='utf-8')

for label_file in label_files:
    img_file = label_file.replace('labels', 'images').replace('txt', 'jpg')
    img = cv2.imread(img_file)
    shape = img.shape
    f = open(label_file, mode='r', encoding='utf-8')
    image_id = int(label_file.split('/')[-1].split('.')[0])
    box = [x.split() for x in f.read().strip().splitlines() if len(x)]
    box = np.array(box, dtype=np.float32)    # (cls, 中心x, 中心y, w, h)
    box[:, 1] *= shape[1]  
    box[:, 3] *= shape[1]  

    box[:, 2] *= shape[0]  
    box[:, 4] *= shape[0]  

    box[:, 1:3] -= (box[:, 3:] / 2)  # xy center to top-left corner

    for b in box.tolist():
        result['annotations'].append(
            {
                'id': cnt,
                'iscrowd': 0,
                'area': b[3] * b[4],
                'image_id': image_id,
                'category_id': int(b[0]),
                'bbox': [round(x, 3) for x in b[1:]]
            }
        )
        cnt += 1
    
    result['images'].append({
        'id': image_id,
    })

json.dump(result, fp)
fp.close()

anno_json = 'result.json'  # annotations json
pred_json = 'runs/train/yolov5m630/predictions.json'

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

anno = COCO(anno_json)  # init annotations api
print('--------------------')
pred = anno.loadRes(pred_json)  # init predictions api
print('--------------------')
eval = COCOeval(anno, pred, 'bbox')
print('--------------------')
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)