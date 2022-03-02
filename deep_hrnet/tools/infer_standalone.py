"""
Standalone inference engine 
(only supports ONNX and gaussian heatmap outputs now)
"""
import torch
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod


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
    
    def __init__(self, input_shape, data_type="coco"):
        self.input_shape = input_shape
        
        try:
            self.skeleton = UdpPsaPoseAbs.SKELETONS[data_type]
        except KeyError:
            self.skeleton = None
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
    @staticmethod    
    def get_max_preds(batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals
    
    def draw_keypoints(self, image, keypoints, radius=1):
        if len(keypoints) == 0:
            return image
        for kpts in keypoints:
            kpts = [list(map(int, x)) for x in kpts]
            if self.skeleton:
                for kid1, kid2 in self.skeleton:
                    cv2.line(image, tuple(kpts[kid1-1]), tuple(kpts[kid2-1]), (0, 255, 0), 2, cv2.LINE_AA)
            # i = 0
            for x, y in kpts:
                # if i >= 13: continue
                # i += 1
                cv2.circle(image, (x, y), radius, (255, 0, 0), 2, cv2.LINE_AA)
        return image
        
    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # normalizes:
        img /= 255.0
        img[:, :, 0] = (img[:, :, 0] - self.mean[0]) / self.std[0]
        img[:, :, 1] = (img[:, :, 1] - self.mean[1]) / self.std[1]
        img[:, :, 2] = (img[:, :, 2] - self.mean[2]) / self.std[2]
        return img.transpose(2, 0, 1)
    
    def _postprocess(self, heatmaps):
        preds, maxvals = UdpPsaPoseAbs.get_max_preds(heatmaps)
        return preds, maxvals
    
    @abstractmethod
    def infer_pose(self, person_crop_image):
        pass

    
class UdpPsaPoseOnnx(UdpPsaPoseAbs):
    
    def __init__(self, model_path, input_shape, data_type="coco"):
        super(UdpPsaPoseOnnx, self).__init__(input_shape, data_type)
        
        import onnxruntime
        
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
    
    @torch.no_grad()
    def infer_pose(self, img):
        pose_input = self._preprocess(img)
        pose_input = np.array(pose_input)[None]
        output = self.ort_session.run(None, {self.input_name: pose_input})[0]
        keypoints, maxvals = self._postprocess(output)
        return keypoints, maxvals, output.shape


if __name__ == "__main__":
    file = "standalone_pose.jpg"
    input_shape = (192, 256)
    engine = UdpPsaPoseOnnx("weights/shufflenetv2plus_pixel_shuffle_256x192_small.onnx",
                            input_shape, "coco")
    img = cv2.imread(file)
    keypoints, maxvals, output_shape = engine.infer_pose(img.copy())
    print(keypoints)
    print(maxvals)
    print(output_shape)
    keypoints[:, :, 0] *= (img.shape[1]/output_shape[3])
    keypoints[:, :, 1] *= (img.shape[0]/output_shape[2])
    img = engine.draw_keypoints(img, keypoints)
    cv2.imshow("Test", img)
    cv2.waitKey()
    