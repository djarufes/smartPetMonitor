# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import os.path
import sys
from pathlib import Path
import numpy as np

import cv2
import csv
from time import time  
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    #print(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Code inserted here:
    # user selection for food and water bowl:
    boxes = []
    # Read image
    img = cv2.VideoCapture(source)
    ret, frame = img.read()
    # frame = cv2.resize(frame, None, fx = 0.3045,fy = 0.5778)

    while(len(boxes) < 2):
        frame_cpy = frame.copy()
        if(len(boxes) == 0): 
            im0 = cv2.rectangle(frame_cpy, (0,0), (600, 80) , (0,0,0), -1)
            im0 = cv2.putText(frame_cpy, 'Select Water Bowl', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            print('Water Bowl Selection')
        elif(len(boxes) == 1): 
            im0 = cv2.rectangle(frame_cpy, (0,0), (570, 80) , (0,0,0), -1)
            im0 = cv2.putText(frame_cpy, 'Select Food Bowl', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
            print('Food Bowl Selection')
        # Select ROI
        r = cv2.selectROI(frame_cpy, fromCenter=False, showCrosshair=False) # returns (x, y, w, h)
        # Bounded image
        boxes.append(((int(r[0]),int(r[1])), (int(r[0]+r[2]),int(r[1]+r[3]))))
        print(boxes[-1])
        cv2.waitKey(1)
    cv2. destroyAllWindows()

    # frame.shape[0] = height and frame.shape[1] = width
    '''
    normBoxes = ((
        (boxes[0][0][0] / frame.shape[1], boxes[0][0][1] / frame.shape[0]), 
        (boxes[0][1][0] / frame.shape[1], boxes[0][1][1] / frame.shape[0])),
        ((boxes[1][0][0] / frame.shape[1], boxes[1][0][1] / frame.shape[0]), 
        (boxes[1][1][0] / frame.shape[1], boxes[1][1][1] / frame.shape[0])))

    normBoxes = ((((boxes[0][0][0] + boxes[0][1][0]) / (2 * frame.shape[1])), ((boxes[0][0][1] + boxes[0][1][1]) / (2 * frame.shape[0])), (boxes[0][1][0] - boxes[0][0][0]) / frame.shape[1], (boxes[0][1][1] - boxes[0][0][1]) / frame.shape[0]),
                 (((boxes[1][0][0] + boxes[1][1][0]) / (2 * frame.shape[1])), ((boxes[1][0][1] + boxes[1][1][1]) / (2 * frame.shape[0])), (boxes[1][1][0] - boxes[1][0][0]) / frame.shape[1], (boxes[1][1][1] - boxes[1][0][1]) / frame.shape[0]))
    print(normBoxes)
    '''
    #inserted code ends here.

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    prev_seen, prev_start_point, prev_end_point = 0, None, None
    activity = None
    timestamp = 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            iou = 0
            unNorm_xyxy = [[0,0,0,0]]
            iou_water, iou_food = 0, 0
            if len(det):
                # Rescale boxes from img_size to im0 size

                ######### code added here:
                # save bounding box info to a varaible.
                #print(det[:, :4])

                ###### code ends here:

                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        unNorm_xyxy = torch.tensor(xyxy).view(1, 4) # Pixel values of cat bounding box upper corner x & y, lower corner x & y
                        print(unNorm_xyxy)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        ### code added here:
                        #print(xywh[1])
                        #print(cls.item())
                        #print(line[1], line[2], line[3], line[4])
                        bounding_box_info = cls.item(), xywh
                        #print(bounding_box_info)

                        # iou calculations:
                        # bounding box info for the cat:
                        x1 = unNorm_xyxy[0][0]
                        x2 = unNorm_xyxy[0][2]
                        y1 = unNorm_xyxy[0][1]
                        y2 = unNorm_xyxy[0][3]
                        #print(x1,y1)
                        #print(x2,y2)
                        area_cat_box = (x2-x1)*(y2-y1)
                        width_waterBowl = boxes[0][1][0]-boxes[0][0][0]
                        height_waterBowl = boxes[0][1][1]-boxes[0][0][1]
                        area_water_box = width_waterBowl * height_waterBowl
                        xleft_water = max(x1,boxes[0][0][0])
                        xright_water = min(x2, boxes[0][1][0])
                        ytop_water = max(y1, boxes[0][0][1])
                        ybottom_water = min(y2, boxes[0][1][1])
                        #print(xleft, xright, ytop, ybottom)
                        if (xright_water<xleft_water) or (ybottom_water<ytop_water):
                          area_inter_water = 0
                        else:
                          area_inter_water = (xright_water-xleft_water)*(ybottom_water-ytop_water)
                        iou_water = area_inter_water/(area_cat_box+area_water_box-area_inter_water)


                        width_foodBowl = boxes[1][1][0]-boxes[1][0][0]
                        height_foodBowl = boxes[1][1][1]-boxes[1][0][1]
                        area_food_box = width_foodBowl * height_foodBowl
                        xleft_food = max(x1,boxes[1][0][0])
                        xright_food = min(x2, boxes[1][1][0])
                        ytop_food = max(y1, boxes[1][0][1])
                        ybottom_food = min(y2, boxes[1][1][1])
                        #print(xleft, xright, ytop, ybottom)
                        if (xright_food<xleft_food) or (ybottom_food<ytop_food):
                          area_inter_food = 0
                        else:
                          area_inter_food = (xright_food-xleft_food)*(ybottom_food-ytop_food)
                        iou_food = area_inter_food/(area_cat_box+area_food_box-area_inter_food)


                        #print(('%g ' * len(line)).rstrip() % line + '\n')'''

                        #normBoxCenter = ((normBoxes[0][0][0] + normBoxes[0][1][0]) / 2, (normBoxes[0][0][1] + normBoxes[0][1][1]) / 2 )
                        #eucledian = ((bounding_box_info[1][0] - normBoxCenter[0])**2 + (bounding_box_info[1][1] - normBoxCenter[1])**2)**(1/2)
                        #print(eucledian)
                        ### code ends here:
                        print(iou_water, iou_food)

                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[0] if hide_conf else f'{names[0]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(0, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                start_point = boxes[0][0]
                end_point = boxes[0][1]
                color = (255, 0, 0)
                thickness = 2
                im0 = cv2.rectangle(im0, (boxes[0][0][0]-1,boxes[0][0][1]-35), (boxes[0][0][0]+140,boxes[0][0][1]) , (255,0,0), -1)
                im0 = cv2.putText(im0, 'Water Bowl', (boxes[0][0][0]+5,boxes[0][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                im0 = cv2.rectangle(im0, start_point, end_point, color, thickness)
                start_point = boxes[1][0]
                end_point = boxes[1][1]
                im0 = cv2.rectangle(im0, (boxes[1][0][0]-1,boxes[1][0][1]-35), (boxes[1][0][0]+130,boxes[1][0][1]) , (255,0,0), -1)
                im0 = cv2.putText(im0, 'Food Bowl', (boxes[1][0][0]+5,boxes[1][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                im0 = cv2.rectangle(im0, start_point, end_point, color, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (950, 120)
                fontScale = 2
                thickness = 2
                color = (0, 255, 255)
                if (iou_water or iou_food): im0 = cv2.putText(im0, 'Audio Model: On', org, font, fontScale, color, thickness, cv2.LINE_AA)
                else: im0 = cv2.putText(im0, 'Audio Model: OFF', org, font, fontScale, color, thickness, cv2.LINE_AA)

                #if(iou_water > iou_food):  
                #elif(iou_food > iou_water): im0 = cv2.putText(im0, 'Eating', org, font, fontScale, color, thickness, cv2.LINE_AA)
                
                # Activity Index
                frames = 15
                if unNorm_xyxy != None:
                    color = (255, 255, 0)
                    perc = 0.10
                    boundingBox_center_x = (int(unNorm_xyxy[0][0]) + int(unNorm_xyxy[0][2])) / 2
                    boundingBox_center_y = (int(unNorm_xyxy[0][1]) + int(unNorm_xyxy[0][3])) / 2
                    boundingBox_width = int(unNorm_xyxy[0][2]) - int(unNorm_xyxy[0][0])
                    boundingBox_height = int(unNorm_xyxy[0][3]) - int(unNorm_xyxy[0][1])
                    activityBox_x1 = int(boundingBox_center_x - (boundingBox_width * perc))
                    activityBox_y1 = int(boundingBox_center_y - (boundingBox_height * perc))
                    activityBox_x2 = int(boundingBox_center_x + (boundingBox_width * perc))
                    activityBox_y2 = int(boundingBox_center_y + (boundingBox_height * perc))
                    start_point = (activityBox_x1, activityBox_y1)
                    end_point = (activityBox_x2, activityBox_y2)

                    # Lagging activityBox
                    if seen - prev_seen > frames:
                        if prev_start_point != prev_end_point:
                            #########
                            # IOU between lagging & real-time activityBox
                            # http://ronny.rest/tutorials/module/localization_001/iou/
                            #########
                            x1 = max(start_point[0], prev_start_point[0])
                            y1 = max(start_point[1], prev_start_point[1])
                            x2 = min(end_point[0], prev_end_point[0])
                            y2 = min(end_point[1], prev_end_point[1])

                            # AREA OF OVERLAP - Area where the boxes intersect
                            width = (x2 - x1)
                            height = (y2 - y1)
                            # handle case where there is NO overlap
                            area_overlap = width * height

                            # COMBINED AREA
                            area_a = (end_point[0] - start_point[0]) * (end_point[1] - start_point[1])
                            area_b = (prev_end_point[0] - prev_start_point[0]) * (prev_end_point[1] - prev_start_point[1])
                            area_combined = area_a + area_b - area_overlap

                            # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
                            iou = 0
                            if area_combined > 0:
                                iou = area_overlap / area_combined
                                print('----------', area_a, area_b, area_combined, area_overlap, iou)
                            
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            org = (950, 220)
                            fontScale = 2
                            if iou > 0: activity = 0    # low
                            else: activity = 1          # high

                        # Update lagging box
                        prev_start_point = start_point
                        prev_end_point = end_point
                        prev_seen = seen

                        ### CSV File Export
                        file_exists = os.path.exists('metaData.csv')
                        header = ['system time', 'timestamp', 'eat_drink', 'activity']
                        data = [datetime.now(),timestamp, iou_food+iou_water, activity]
                        timestamp += 0.5
                        with open('metaData.csv', 'a', encoding='UTF8') as f:
                            writer = csv.writer(f)
                            # If file does not exists, write the header
                            if file_exists == False: writer.writerow(header)
                            writer.writerow(data)

                    elif prev_start_point != prev_end_point:
                        color = (255, 0, 255)
                        im0 = cv2.rectangle(im0, prev_start_point, prev_end_point, color, thickness)
                    # Real-time activityBox
                    color = (255, 255, 0)
                    im0 = cv2.rectangle(im0, start_point, end_point, color, thickness)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (950, 220)
                    fontScale = 2
                    if activity == 0: im0 = cv2.putText(im0, 'Activity: Low', org, font, fontScale, (0, 255, 255), 2, cv2.LINE_AA)
                    elif activity == 1: im0 = cv2.putText(im0, 'Activity: High', org, font, fontScale, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)