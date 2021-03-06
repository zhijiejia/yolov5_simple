python train.py \
--data hw.yaml \
--img 640 \
--name yolov5m6 \
--weights yolov5m6.pt \
--batch-size 16 \
--cos-lr \
--multi-scale \
--device 2 \
--epoch 80 \
--save-period 1 \
--hyp data/hyps/hyp.scratch-high.yaml \