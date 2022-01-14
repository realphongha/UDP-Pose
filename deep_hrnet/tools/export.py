import sys
import os
from pathlib import Path

import torch

import argparse

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
sys.path.insert(0, str(os.path.join(ROOT, "lib")))

from lib.config import cfg as config
from lib.models import MODELS


def export_onnx(model, img, opt):
    import onnx

    torch.onnx.export(model, img, opt.file, 
                      verbose=False, 
                      opset_version=opt.opset,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes={'images': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    # Checks
    model_onnx = onnx.load(opt.file)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # print(onnx.helper.printable_graph(model_onnx.graph))  # print

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(opt.file)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(model(img)), ort_outs[0], rtol=1e-03, atol=1e-05)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_model', type=str, required=True, help='path to pretrained pose model')
    parser.add_argument('--cfg', type=str, required=True, help='path to pose model configuration file')
    parser.add_argument('--format', type=str, default="onnx", help='format to export')
    parser.add_argument('--file', type=str, required=True, help='filename to export')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
    opt = parser.parse_args()
    return opt

def main(opt):
    device = torch.device(opt.device)
    
    # update config
    config.defrost()
    config.merge_from_file(opt.cfg)  
    pose_model = MODELS[config.MODEL.NAME](config, is_train=False)
    state_dict = torch.load(opt.pose_model, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module." in k:
            name = k[7:]  # remove "module"
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict
    pose_model.load_state_dict(state_dict, strict=False)
    pose_model.to(device)
    pose_model.eval()
    
    imgsz = config.MODEL.IMAGE_SIZE
    img = torch.zeros(opt.batch, 3, imgsz[1], imgsz[0]).to(device)
    if opt.format == "onnx":
        export_onnx(pose_model, img, opt)
    else:
        raise Exception("%s format is not supported!" % opt.format)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
