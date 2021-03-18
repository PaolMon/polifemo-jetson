import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import sys, os
sys.path.insert(1, './yolo')

from yolo.onnx_to_tensorrt import get_engine, draw_bboxes
from yolo.data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

#from capture_frame import capture_frame, gstreamer_pipeline


sys.path.insert(2, '/usr/src/tensorrt/samples/python')
import common
import argparse


import cv2
from PIL import Image
import time

dispW=416
dispH=416
flip=2


def detect():

    args = parse_args()
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = 'yolo/%s.onnx' % args.model
    engine_file_path = 'yolo/%s.trt' % args.model

    # Download a dog image and save it to the following file path:
    # input_image_path = 'frame.png'
    camSet='nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 !  video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.15 saturation=1.2 ! appsink drop=true'
    cap= cv2.VideoCapture(camSet)
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    # input_resolution_yolov3_HW = (608, 608)
    input_resolution_yolov3_HW = (w, h)

    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    
    

    # Output shapes expected by the post-processor
    #output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
    output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26)]

    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        

        #postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    # A list of 3 three-dimensional tuples for the YOLO masks
        postprocessor_args = {"yolo_masks": [(3, 4, 5), (0, 1, 2)],                    # A list of 2 three-dimensional tuples for the YOLOv3-tiny-416 masks
                            "yolo_anchors": [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)],  
        #                      "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  # for yolov3-608
        #                                       (59, 119), (116, 90), (156, 198), (373, 326)],
                            "obj_threshold": 0.5,                                               # Threshold for object coverage, float value between 0 and 1
                            "nms_threshold": 0.5,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                            "yolo_input_resolution": input_resolution_yolov3_HW}

        postprocessor = PostprocessYOLO(**postprocessor_args)

        while cap.isOpened():

            t0 = time.time()

            hf, im_bgr = cap.read()
            im_rgb = im_bgr[:, :, [2, 1, 0]]
            if hf:
                input_image_path = Image.fromarray(im_rgb, 'RGB')


            # Load an image from the specified input path, and return it together with  a pre-processed version
            image_raw, image = preprocessor.process(input_image_path)
            # Store the shape of the original input image in WH format, we will need it for later
            shape_orig_WH = image_raw.size
            
            # Do inference

            # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
            inputs[0].host = image

            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]


            # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
            boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))

            print('###################################')

            if boxes is not None:
                print('FPS: {}'.format(1/(time.time() - t0)))

            
                # Draw the bounding boxes onto the original input image and save it as a PNG file
                #obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
                #output_image_path = 'frame_bboxes.png'
                #obj_detected_img.save(output_image_path, 'PNG')
                #print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
            
            cv2.imshow('yolofemo',im_bgr)
            cv2.moveWindow('yolofemo', 0, 25)
            if cv2.waitKey(1)==ord('q'):
                break

def parse_args():
    """Parse input arguments."""
    desc = ('Capture live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #capture_frame()
    detect()
