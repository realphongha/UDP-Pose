python inference_engine.py --imgsz 640 --source 0 --device cpu ^
 --det_model yolov5s.pt --conf-thres 0.4 --iou-thres 0.3 ^
 --cfg deep_hrnet\experiments\coco\shufflenetv2\10x_256x192_adam_lr1e-3_pixel_shuffle.yaml ^
 --pose_model E:\Learning\do_an_tot_nghiep\experiments\pose\pose_shufflenetv2_10x_pixel_shuffle\10x_256x192_adam_lr1e-3_pixel_shuffle\model_best.onnx ^
 --pose_format onnx ^
 --person_class 0 ^
 --classes 1 2 3 4