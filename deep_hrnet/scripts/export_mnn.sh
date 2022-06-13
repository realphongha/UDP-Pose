python tools/export.py \
 --pose_model E:\Learning\do_an_tot_nghiep\experiments\pose\pose_shufflenetv2_plus_pixel_shuffle\small_256x192_adam_lr1e-3_pixel_shuffle\model_best.pth \
 --cfg experiments\coco\shufflenetv2+\small_256x192_adam_lr1e-3_pixel_shuffle.yaml \
 --format onnx \
 --file weights/shufflenetv2plus_pixel_shuffle_256x192_small.onnx \
 --device cpu \
 --batch 1 \
 --opset 11
mnnconvert -f ONNX \
            --modelFile weights/shufflenetv2plus_pixel_shuffle_256x192_small.onnx \
            --MNNModel weights/shufflenetv2plus_pixel_shuffle_256x192_small.mnn