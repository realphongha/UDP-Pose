python deep_hrnet/tools/export.py ^
 --pose_model E:\Learning\do_an_tot_nghiep\experiments\pose\pose_shufflenetv2_10x_pixel_shuffle\10x_256x192_adam_lr1e-3_pixel_shuffle\model_best.pth ^
 --cfg deep_hrnet\experiments\coco\shufflenetv2\10x_256x192_adam_lr1e-3_pixel_shuffle.yaml ^
 --format onnx ^
 --file E:\Learning\do_an_tot_nghiep\experiments\pose\pose_shufflenetv2_10x_pixel_shuffle\10x_256x192_adam_lr1e-3_pixel_shuffle\model_best.onnx ^
 --device cpu ^
 --batch 1 ^
 --opset 12 ^