python deep_hrnet/tools/export.py \
 --pose_model path/to/model.pth \
 --cfg deep_hrnet/experiments/coco/shufflenetv2/10x_256x192_adam_lr1e-3_pixel_shuffle.yaml \
 --format onnx \
 --file output/path/model.onnx \
 --device cpu \
 --batch 1 \
 --opset 12
python -m onnxruntime.tools.convert_onnx_models_to_ort \
 output/path/model.onnx