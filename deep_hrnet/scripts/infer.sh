python inference_engine.py --imgsz 640 \
 --source path/to/data_source \
 --device 0 \
 --det_model yolov5s.pt --conf-thres 0.4 --iou-thres 0.3 \
 --cfg deep_hrnet\experiments\coco\hrnet\w32_256x192_adam_lr1e-3_offset_ofm_psa.yaml \
 --pose_model path/to/pose_model.pth \
 --pose_format torch \
 --fps 5 \
 --person_class 0 \
 --bbox-dir path/to/bbox_dir \
 --save-dir path/to/save/pose_label
#  --classes 1 2 3 4