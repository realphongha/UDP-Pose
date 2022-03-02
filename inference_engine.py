from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from re import L

import sys
import os
from black import out
from tqdm import tqdm
from pathlib import Path
from time import time
from abc import ABCMeta, abstractmethod


import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as T

import argparse

import cv2
import numpy as np

from deep_hrnet.tools.infer_utils.boxes import letterbox, scale_boxes, non_max_suppression
from deep_hrnet.tools.infer_utils.boxes import yolo2xyxy
from deep_hrnet.tools.infer_utils.utils import WebcamStream

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
sys.path.append(str(os.path.join(ROOT, "yolov5")))

from models.experimental import attempt_load
from models.yolo import Model

ROOT2 = os.path.join(ROOT, "deep_hrnet")
if str(ROOT2) not in sys.path:
    sys.path.append(str(ROOT2))  # add ROOT to PATH
sys.path.insert(0, str(os.path.join(ROOT2, "lib")))
from lib.models import MODELS
from lib.core.inference import get_final_preds
from deep_hrnet.pose_engine import *


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_model', type=str, default=None,
                        help='path to pretrained pose model')
    parser.add_argument('--pose_format', type=str,
                        default='torch', help='pose model format')
    parser.add_argument('--cfg', type=str, required=True,
                        help='path to pose model configuration file')
    parser.add_argument('--det_model', nargs='+', type=str,
                        default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--weights', type=str, default=None,
                        help='.pth weights for det_model')
    parser.add_argument('--source', nargs='+', type=str, default=ROOT /
                        'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--padding', type=int, default=5,
                        help='Human bounding box padding for pose estimation')
    parser.add_argument('--fps', type=int, default=None,
                        help='FPS for output video')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--pose-device', default='cpu', help='device for pose estimation')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--person_class', default=0,
                        type=int, help='person class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default=ROOT /
                        'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--bbox-dir', default=None,
                        help='bbox detection results for pose labelling')
    parser.add_argument('--save-dir', default=None,
                        help='path to save pose detection results')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3,
                        type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False,
                        action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False,
                        action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--save-lbl', action='store_true',
                        help='save detection and pose labels')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


class YoloDetectionAbs(metaclass=ABCMeta):

    def __init__(self, opt):
        self.input_shape = opt.imgsz
        self.model_path = opt.det_model
        self.opt = opt
        self.padding = opt.padding
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.classes = opt.classes
        if self.classes is None:
            self.classes = list()
        self.classes.append(opt.person_class)

    @staticmethod
    def padding_bbox(x1, y1, x2, y2, img_shape):
        h, w = img_shape[:2]
        x1 -= 5
        y1 -= 5
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 += 5
        y2 += 5
        x2 = w if x2 > w else x2
        y2 = h if y2 > h else y2
        return x1, y1, x2, y2

    @abstractmethod
    def _preprocess(self, image):
        # return model input
        pass

    @abstractmethod
    def _postprocess(self, heatmaps):
        # return boxes
        pass

    @abstractmethod
    def infer(self, image):
        pass


class YoloDetectionTorch(YoloDetectionAbs):

    def __init__(self, opt, device):
        super(YoloDetectionTorch, self).__init__(opt)
        self.device = device
        if opt.weights:
            num_classes = 1 if (opt.classes is None) else (len(opt.classes)+1)
            self.model = Model(opt.det_model[0], ch=3, nc=num_classes)
            self.model.load_state_dict(torch.load(
                opt.weights, map_location=device)["model_state_dict"])
        else:
            self.model = attempt_load(
                opt.det_model, map_location=device, fuse=True)

        self.model = self.model.to(device)
        self.model.eval()

    def _preprocess(self, image):
        img = letterbox(image, new_shape=self.input_shape)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def _postprocess(self, pred, img, raw_img):
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   classes=self.classes,
                                   max_det=self.opt.max_det)
        det = pred[0]

        if len(det):
            person_boxes = list()
            boxes = scale_boxes(
                det[:, :4], raw_img.shape[:2], img.shape[-2:]).cpu()
            for i, box in enumerate(boxes.numpy()):
                box_class = int(det[i][5])
                if box_class != opt.person_class and box_class not in opt.classes:
                    continue
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, y1, x2, y2 = YoloDetectionAbs.padding_bbox(
                    x1, y1, x2, y2, raw_img.shape)
                if box_class == opt.person_class:
                    person_boxes.append([x1, y1, x2, y2])
                raw_img = cv2.rectangle(
                    raw_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(raw_img, str(box_class), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
            if person_boxes:
                return torch.Tensor(person_boxes)
            else:
                return None
        return None

    def infer(self, image):
        img = self._preprocess(image)
        pred = self.model(img)[0]
        boxes = self._postprocess(pred, img, image)
        return boxes


def main(opt):
    device = torch.device(opt.device)

    det_engine = YoloDetectionTorch(opt, device)
    if opt.pose_format == 'torch':
        pose_engine = UdpPsaPoseTorch(opt.pose_model,
                                      opt.cfg,
                                      device)
    elif opt.pose_format == 'onnx':
        pose_engine = UdpPsaPoseOnnx(opt.pose_model,
                                     opt.cfg,
                                     opt.device)
    elif opt.pose_format == 'openvino':
        pose_engine = UdpPsaPoseOpenVino(opt.pose_model,
                                         opt.cfg,
                                         opt.pose_device)
    elif opt.pose_format == 'mnn':
        pose_engine = UdpPsaPoseMNN(opt.pose_model,
                                    opt.cfg,
                                    opt.pose_device)
    else:
        raise Exception("%s format is not implemented!" % opt.pose_format)

    for s in opt.source:
        print("Processing %s ..." % s)
        source = Path(s)
        if source.is_file() and source.suffix in ['.jpg', '.jpeg', '.png']:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR |
                               cv2.IMREAD_IGNORE_ORIENTATION)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = det_engine.infer(image)
            if boxes is not None:
                keypoints, _ = pose_engine.infer_pose(image, boxes)
                output = pose_engine.draw_keypoints(image, keypoints, radius=1)
            cv2.imwrite(f"{str(source).rsplit('.', maxsplit=1)[0]}_out.jpg", cv2.cvtColor(
                output, cv2.COLOR_RGB2BGR))

        elif source.is_dir():
            files = source.glob("*.jpg")
            # print(list(files))
            many_dirs = False
            if not list(files):
                many_dirs = True
                sources = []
                dirs = source.glob("*")
                for d in dirs:
                    sources.append(Path(d))
            else:
                sources = [source]
            for source in sources:
                files = source.glob("*.jpg")
                for file in tqdm(files):
                    image = cv2.imread(str(file), cv2.IMREAD_COLOR |
                                    cv2.IMREAD_IGNORE_ORIENTATION)
                    img_h, img_w = image.shape[:2]
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if opt.bbox_dir:
                        dir_path, fn = os.path.split(file)
                        if many_dirs:
                            bbox_dir = os.path.join(opt.bbox_dir, os.path.split(dir_path)[1])
                        else:
                            bbox_dir = opt.bbox_dir
                        output_path = os.path.join(opt.save_dir, os.path.split(dir_path)[1])
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)
                        lbl_path = os.path.join(bbox_dir, fn.replace(".jpg", ".txt"))
                        lbl_file = open(lbl_path, "r")
                        lbl = lbl_file.read().splitlines()[0]
                        yolo_bbox = list(map(float, lbl.strip().split()[1:]))
                        bbox = yolo2xyxy(image.shape[:2], yolo_bbox)
                        image = cv2.rectangle(image, (bbox[0], bbox[1]), 
                                            (bbox[2], bbox[3]), (0, 0, 0))
                        keypoints, maxvals = pose_engine.infer_pose(image, torch.Tensor(np.array(bbox)[None]))
                        output = pose_engine.draw_keypoints(
                            image, keypoints, radius=1)
                        keypoint = keypoints[0][:13]
                        maxvals = maxvals[0][:13]
                        # print(keypoint);print(maxvals);quit()
                        with open(os.path.join(output_path, fn.replace(".jpg", ".txt")), "w") as pose_file:
                            for i, k in enumerate(keypoint):
                                pose_file.write("%f %f %f\n" % (k[0]/img_w, k[1]/img_h, maxvals[i][0]))
                        # cv2.imshow("Result", output) 
                        # cv2.waitKey()
                        # quit()
                    else:
                        boxes = det_engine.infer(image)
                        if boxes is not None:
                            keypoints, _ = pose_engine.infer_pose(image, boxes)
                            output = pose_engine.draw_keypoints(
                                image, keypoints, radius=1)
                        # cv2.imshow("Result", cv2.cvtColor(
                        #     output, cv2.COLOR_RGB2BGR)) 
                        # cv2.waitKey()
                        cv2.imwrite(f"{str(file).rsplit('.', maxsplit=1)[0]}_out.jpg", cv2.cvtColor(
                            output, cv2.COLOR_RGB2BGR))
                    # break

        elif source.is_file() and source.suffix in ['.mp4', '.avi', '.mkv']:
            video_out = f"{s.rsplit('.', maxsplit=1)[0]}_out.mp4"
            video_reader = cv2.VideoCapture(s)

            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            if opt.fps:
                fps = opt.fps
            else:
                fps = int(video_reader.get(cv2.CAP_PROP_FPS))

            video_writer = cv2.VideoWriter(video_out,
                                           cv2.VideoWriter_fourcc(*'MPEG'),
                                           fps,
                                           (frame_w, frame_h))

            pose_fps = []
            for i in tqdm(range(nb_frames)):
                begin_fps = time()
                ret, frame = video_reader.read()
                if not ret:
                    break
                if frame is None:
                    continue
                boxes = det_engine.infer(frame)
                if boxes is not None:
                    begin_pose_fps = time()
                    keypoints, _ = pose_engine.infer_pose(frame, boxes)
                    pose_fps.append((time()-begin_pose_fps)/keypoints.shape[0])
                    output = pose_engine.draw_keypoints(
                        frame, keypoints, radius=1)
                pose_pps = 1/np.mean(pose_fps) if pose_fps else -1
                fps = 1/(time()-begin_fps)
                cv2.putText(frame, "FPS: %.2f, Pose PPS: %.2f" % (fps, pose_pps), (0, frame_h-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                video_writer.write(np.uint8(frame))
                if i % 100 == 0:
                    print("Pose FPS:", 1/np.mean(pose_fps))

            print("Pose FPS:", 1/np.mean(pose_fps))
            video_reader.release()
            video_writer.release()
        else:
            webcam = WebcamStream()

            i = 1
            fps = []
            pose_fps = []
            for frame in webcam:
                begin_fps = time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = det_engine.infer(frame)
                if boxes is not None:
                    begin_pose_fps = time()
                    keypoints, _ = pose_engine.infer_pose(frame, boxes)
                    pose_fps.append(time()-begin_pose_fps)
                    frame = pose_engine.draw_keypoints(
                        frame, keypoints, radius=1)
                fps.append(time()-begin_fps)
                
                if i % 10 == 0:
                    if fps:
                        print("FPS:", 1/np.mean(fps))
                    if pose_fps:
                        print("Pose FPS:", 1/np.mean(pose_fps))
                i += 1
                cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
