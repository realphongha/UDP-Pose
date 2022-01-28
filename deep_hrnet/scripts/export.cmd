python deep_hrnet/tools/export.py ^
 --pose_model E:\Workspace\git\UDP-Pose\deep_hrnet\weights\hrnet_w32_256x192.pth ^
 --cfg deep_hrnet\experiments\coco\hrnet\w32_256x192_adam_lr1e-3_offset_ofm_psa.yaml ^
 --format onnx ^
 --file E:\Workspace\git\UDP-Pose\deep_hrnet\weights\hrnet_w32_256x192.onnx ^
 --device cpu ^
 --batch 1 ^
 --opset 12 ^