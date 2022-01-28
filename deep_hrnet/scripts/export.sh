python deep_hrnet/tools/export.py \
 --pose_model path/to/input_model.pth \
 --cfg path/to/cfg.yaml \
 --format onnx \
 --file path/to/output_model.onnx \
 --device cpu \
 --batch 1 \
 --opset 12 \