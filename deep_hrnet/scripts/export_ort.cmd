python deep_hrnet/tools/export.py ^
 --pose_model E:\Learning\do_an_tot_nghiep\experiments\pose\pose_shufflenetv2_plus\small_256x192_adam_lr1e-3\model_best.pth ^
 --cfg E:\Learning\do_an_tot_nghiep\experiments\pose\pose_shufflenetv2_plus\small_256x192_adam_lr1e-3\small_256x192_adam_lr1e-3.yaml ^
 --format onnx ^
 --file E:\Learning\do_an_tot_nghiep\experiments\pose\converted\pose_shufflenetv2_plus.onnx ^
 --device cpu ^
 --batch 1 ^
 --opset 12
python -m onnxruntime.tools.convert_onnx_models_to_ort ^
 E:\Learning\do_an_tot_nghiep\experiments\pose\converted\pose_shufflenetv2_plus.onnx