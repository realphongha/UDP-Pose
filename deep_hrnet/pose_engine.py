import torch
import cv2
import numpy as np

from abc import ABCMeta, abstractmethod

import torchvision.transforms as T

from .lib.models import MODELS
from .lib.core.inference import get_final_preds
from .tools.infer_utils.utils import get_affine_transform
from .tools.infer_utils.utils import draw_keypoints


class UdpPsaPoseAbs(metaclass=ABCMeta):
    
    SKELETONS = {"coco":[
                    [16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13], 
                    [6,7], [6,8], [7,9], [8,10], [9,11], [2,3], [1,2], [1,3], [2,4], 
                    [3,5], [4,6], [5,7]
                ],
                "mpii": [
                    [9, 10], [12, 13], [12, 11], [3, 2], [2, 1], [14, 15], 
                    [15, 16], [4, 5], [5, 6], [9, 8], [8, 7], [7, 3], [7, 4], 
                    [9, 13], [9, 14]
                ]}
    
    def __init__(self, config_path):
        from .lib.config import cfg as config
        
        self.config = config
        self.config.defrost()
        self.config.merge_from_file(config_path)
        self.input_shape = self.config.MODEL.IMAGE_SIZE
        
        try:
            self.skeleton = UdpPsaPoseAbs.SKELETONS[self.config.DATASET.DATASET.lower()]
        except KeyError:
            self.skeleton = None
        self.pose_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
    @staticmethod    
    def xyxy2xywh(x):
        x = torch.tensor(x).float()
        y = x.clone()
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y
        
    def _box_to_center_scale(self, boxes, pixel_std=200):
        boxes = UdpPsaPoseAbs.xyxy2xywh(boxes)
        r = self.input_shape[0] / self.input_shape[1]
        mask = boxes[:, 2] > boxes[:, 3] * r
        boxes[mask, 3] = boxes[mask, 2] / r
        boxes[~mask, 2] = boxes[~mask, 3] * r
        boxes[:, 2:] /= pixel_std 
        boxes[:, 2:] *= 1.25
        return boxes
    
    def draw_keypoints(self, image, keypoints, radius=1):
        draw_keypoints(image, keypoints, self.skeleton, radius)
        return image
        
    def _preprocess(self, img, boxes):
        boxes = self._box_to_center_scale(boxes)
        image_patches = []
        for cx, cy, w, h in boxes:
            trans = get_affine_transform(np.array([cx, cy]), 
                                         np.array([w, h]), 
                                         self.input_shape)
            img_patch = cv2.warpAffine(
                img, 
                trans, 
                (int(self.input_shape[0]), int(self.input_shape[1])), 
                flags=cv2.INTER_LINEAR)
            img_patch = self.pose_transform(img_patch)
            image_patches.append(img_patch)
            
        image_patches = torch.stack(image_patches)
        return image_patches, boxes
    
    def _postprocess(self, outputs, boxes):
        preds, maxvals, preds_in_input_space = \
                        get_final_preds(self.config, outputs, 
                                        boxes[:, :2].numpy(), 
                                        boxes[:, 2:].numpy())
        return preds, maxvals
    
    @abstractmethod
    def infer_pose(self, person_crop_image):
        pass


class UdpPsaPoseTorch(UdpPsaPoseAbs):
    
    def __init__(self, model_path, config_path, device):
        super(UdpPsaPoseTorch, self).__init__(config_path)
        
        self.config.TEST.MODEL_FILE = model_path
        self._device = device
        self.model = MODELS[self.config.MODEL.NAME](self.config, is_train=False)
        state_dict = torch.load(model_path, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module." in k:
                name = k[7:]  # remove "module"
            else:
                name = k
            new_state_dict[name] = v
        state_dict = new_state_dict
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def infer_pose(self, img, boxes):
        pose_input, boxes =  self._preprocess(img, boxes)
        pose_input = pose_input.to(self._device)
        outputs = self.model(pose_input).clone().cpu().numpy()
        keypoints, maxvals = self._postprocess(outputs, boxes)
        return keypoints, maxvals
    
    
class UdpPsaPoseOnnx(UdpPsaPoseAbs):
    
    def __init__(self, model_path, config_path, device):
        super(UdpPsaPoseOnnx, self).__init__(config_path)
        
        import onnxruntime
        
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        
        self._device = device
    
    @torch.no_grad()
    def infer_pose(self, img, boxes):
        pose_input, boxes =  self._preprocess(img, boxes)
        pose_input = np.array(pose_input)
        outputs = None
        for i in range(pose_input.shape[0]):
            output = self.ort_session.run(None, {self.input_name: pose_input[i][None]})[0]
            outputs = output if outputs is None else np.concatenate((outputs, output), axis=0)
        keypoints, maxvals = self._postprocess(outputs, boxes)
        return keypoints, maxvals    


class UdpPsaPoseOpenVino(UdpPsaPoseAbs):
    
    def __init__(self, model_path, config_path, device):
        super(UdpPsaPoseOpenVino, self).__init__(config_path)
        
        from openvino.inference_engine import IECore
        
        ie = IECore()
        net = ie.read_network(model=model_path + ".xml", 
                              weights=model_path + ".bin")
        self._device = device
        self.model = ie.load_network(net, self._device)
        self._outputs = list(self.model.outputs.keys())
        self.input_blob = next(iter(self.model.input_info))
        self.output_blob = next(iter(self.model.outputs))
    
    @torch.no_grad()
    def infer_pose(self, img, boxes):
        pose_input, boxes =  self._preprocess(img, boxes)
        outputs = None
        for i in range(pose_input.shape[0]):
            inference_results = self.model.infer(inputs={self.input_blob: pose_input[i]})
            output = inference_results[self._outputs[0]]
            if outputs is None:
                outputs = output
            else:
                outputs = np.concatenate((outputs, output))
        keypoints, maxvals = self._postprocess(outputs, boxes)
        return keypoints, maxvals    
    
    
class UdpPsaPoseMNN(UdpPsaPoseAbs):
    
    def __init__(self, model_path, config_path, device):
        super(UdpPsaPoseMNN, self).__init__(config_path)

        import MNN
        self.MNNlib = MNN

        self.interpreter = self.MNNlib.Interpreter(model_path)
        # self.interpreter.setCacheFile('.tempcache')
        config = {}
        # config['precision'] = 'low'
        runtimeinfo, exists = self.MNNlib.Interpreter.createRuntime((config,))
        # print(runtimeinfo, exists)
        self.session = self.interpreter.createSession(config, runtimeinfo)
        self.input_tensor = self.interpreter.getSessionInput(self.session, "images")  
    
    @torch.no_grad()
    def infer_pose(self, img, boxes):
        pose_input, boxes =  self._preprocess(img, boxes)
        self.interpreter.resizeTensor(self.input_tensor, pose_input.shape)
        self.interpreter.resizeSession(self.session)
        tmp_input = self.MNNlib.Tensor(
            pose_input.shape,
            self.MNNlib.Halide_Type_Float,
            pose_input.numpy(), 
            self.MNNlib.Tensor_DimensionType_Caffe
        )
        if not self.input_tensor.copyFrom(tmp_input):
            raise Exception("Copying MNN tensor failed!")
        self.interpreter.runSession(self.session)
        raw_outputs = self.interpreter.getSessionOutput(self.session)

        # construct a tmp tensor and copy/convert in case output_tensor is nc4hw4
        # tmp_output = self.MNNlib.Tensor(
        #     raw_outputs.getShape(), 
        #     self.MNNlib.Halide_Type_Float, 
        #     np.ones(raw_outputs.getShape()).astype(np.float32), 
        #     self.MNNlib.Tensor_DimensionType_Caffe)
        # raw_outputs.copyToHostTensor(tmp_output)

        outputs = raw_outputs.getNumpyData()
        keypoints, maxvals = self._postprocess(outputs, boxes)
        return keypoints, maxvals    
