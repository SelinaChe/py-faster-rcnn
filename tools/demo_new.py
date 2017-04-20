#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms #fast-rcnn-master\fast-rcnn-master\lib\utils
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

import rospy
from std_msgs.msg import String

import threading
import time

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES_COCO = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter','bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven','toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush')

NETS = {'vgg16': ('VGG16',
                  'coco_vgg16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'coco': ('COCO',
                  'coco_vgg16_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
#    print(dets)
    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
#    print '**********1***************'
#    print class_name, score
    return class_name, score

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, fc7layer = im_detect(net, im)
    print("boxes: (%d, %d)" % (len(boxes),len(boxes[0])))
    print(boxes)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    count = 0
    class_name_list = []
    score_list = []
    for cls_ind, cls in enumerate(CLASSES_COCO[1:]):
        count = count + 1
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
#        print '*********************'
#        print vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #class_name, score = vis_detections(im, cls, dets, thresh=CONF_THRESH)
        max_box_index = 0.0
        print "dets.shape: (%d, %d)" % (len(dets), len(dets[0]))
#        np.savetxt('real_gan_fc7/boxes.txt', dets)
        for i in range(len(dets)):
            box_size = (dets[i,2]-dets[i,0])**2 + (dets[i,3]-dets[i,1])**2
            if box_size > max_box_index:
                max_box_index = i
#        print dets[max_box_index,-1]
        inds = np.where(dets[:, -1] >= 0.5)[0]
        score = 0.1
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            print "bbox len: %d" % len(bbox)
            print bbox
            print "test01: %d " % i
            print fc7layer[i+1]
        if score >=0.5:

        #print cls_scores[:, np.newaxis]
            class_name_list.append(cls)
            score_list.append(score)
            #np.savetxt("real_gan_fc7/gan_fc7.txt", )
    return class_name_list, score_list

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--image', dest='image_path',
                        help='image full path')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    time1 = time.time()
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0], 
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    time2 = time.time()
    print "*********************************"
    print time2-time1
    print "*********************************"
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
#    for i in xrange(2):
#        _, _= im_detect(net, im)

    #im_name = 'dog0.953836_crop.jpg'
    class_list, score_list = demo(net, args.image_path)
    result = ''
    if len(class_list)<1:
        result = '***'
    for i in range(len(class_list)):
        result = result + class_list[i] + ':' + str(score_list[i])+ ' , '
    print result
#    return result
        

    #plt.show()
'''
def result_pub(pub):
	result = train_and_test()
	pub.publish(result)

if __name__ == "__main__":
	im_name = sys.argv[1]

	pub = rospy.Publisher('image/result',String,queue_size=10)
	rospy.init_node('image_result', anonymous=True)
	dirname = '/home/exbot/catkin_ws/src/image_recognition/src/tools/images/'
	origin = set([_f[2] for _f in os.walk(dirname)][0])
	threads = []
	result_pub(pub)

	while True:
		difference = set([_f[2] for _f in os.walk(dirname)][0]).difference(origin)
		
		if len(difference) >= 1:
			difference = list(difference)
			print difference[0]
			f = os.path.join(dirname, difference[0])
			th = threading.Thread(target = result_pub, args=(pub, ))
			th.start()
			threads.append(th)
		origin = set([_f[2] for _f in os.walk(dirname)][0])
	
'''
