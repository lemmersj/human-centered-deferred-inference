## Pre-processing images for UNITER
- Run move_coco_for_eval.py
- in the bottom-up-attention.pytorch package, run:
`python extract_features.py --mode caffe --num-cpus 32 --gpus '0' --extract-mode bbox_feats --config-file configs/caffe/test-caffe-r101.yaml --image-dir images --bbox-dir boxes --out-dir outdir --resume

## Checking accuracy
python check_acc.py

val: reported: 91.64, reimplemented: 89.87 (-1.77)
testA: reported: 92.26, reimplemented: 91.14 (-1.12)
testB: reported: 90.46, reimplemented: 88.95 (-1.51)
