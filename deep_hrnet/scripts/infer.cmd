python inference_engine.py --imgsz 640 ^
 --source E:\Workspace\data\action\exam\data ^
 --device cpu ^
 --det_model yolov5s.pt --conf-thres 0.4 --iou-thres 0.3 ^
 --cfg deep_hrnet\experiments\coco\hrnet\w32_256x192_adam_lr1e-3_offset_ofm_psa.yaml ^
 --pose_model E:\Workspace\git\UDP-Pose\deep_hrnet\weights\hrnet_w32_256x192.pth ^
 --pose_format torch ^
 --fps 5 ^
 --person_class 0 ^
 --bbox-dir E:\Workspace\data\action\exam\objdet_label\ ^
 --save-dir E:\Workspace\data\action\exam\pose_label
@REM  --classes 1 2 3 4
@REM --pose_model E:\Learning\do_an_tot_nghiep\experiments\pose\pose_shufflenetv2_plus_pixel_shuffle\small_256x192_adam_lr1e-3_pixel_shuffle\model_best.onnx ^