import sys

import _init_carla_paths

from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

import tensorflow as tf
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
# from config import cfg, config
import numpy as np

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1


def vis_detections(im, class_name, dets, thresh=0.5, image_name='demo.jpg'):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('result_{}_{}'.format(class_name, image_name))

class simple_DetBox(object):
    def __init__(self, x=0.0, y=0.0, w=0.0, h=0.0, tag=None, score=0.0):
        self.score=score
        self.x = x
        self.y = y
        self.x1 = x + w
        self.y1 = y + h


faster_model_path = '/home/foxfi/unreal/tf-faster-rcnn/data/voc_2007_trainval+voc_2012_trainval/res101_faster_rcnn_iter_110000.ckpt'
class FasterRCNN(object):
    # Faster RCNN voc demo
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
    DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
    def __init__(self, session, demonet = 'res101'):
        self.session = session
        self.CAR_INDEX = 7
        if demonet == 'vgg16':
            self.net = vgg16()
        elif demonet == 'res101':
            self.net = resnetv1(num_layers=101)
        else:
            raise NotImplementedError
        self.net.create_architecture("TEST", 21, tag='default', anchor_scales=[8, 16, 32])
        self.saver = tf.train.Saver()

    def load(self, model_file=None):
        self.saver.restore(self.session, model_file)
        print('Loaded network {:s}'.format(model_file))

    def get_detbox(self, image, only_car=True):
        scores, boxes = im_detect(self.session, self.net, image)
        pred_boxes = boxes.reshape(-1, len(self.CLASSES), 4)
        result_boxes = []
        # Visualize detections for each class
        CONF_THRESH = 0.0
        NMS_THRESH = 0.5
        test_max_boxes_per_image = 100
        if only_car:
            cls_ind = self.CAR_INDEX
            inds = np.where(scores[:, cls_ind] > CONF_THRESH)[0]
            cls_scores = scores[inds, cls_ind]
            cls_boxes = boxes[inds, 4 * cls_ind:4 * (cls_ind + 1)]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)

            dets = dets[keep, :]
            # dets = np.array(dets[keep, :], dtype=np.float, copy=False)

            for i in range(dets.shape[0]):
                db = dets[i, :]
                dbox = simple_DetBox(
                    db[0], db[1], db[2] - db[0], db[3] - db[1],
                    tag=self.CLASSES[cls_ind], score=db[-1])
                result_boxes.append(dbox)
            if len(result_boxes) > test_max_boxes_per_image:
                result_boxes = sorted(
                    result_boxes, reverse=True, key=lambda t_res: t_res.score) \
                    [:test_max_boxes_per_image]
            # TODO: handle no detect?
            box = result_boxes[0]
            return box
        else:
            # Visualize detections for each class
            for cls_ind, cls in enumerate(self.CLASSES[1:]):
                cls_ind += 1  # because we skipped background
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                vis_detections(image, cls, dets, thresh=CONF_THRESH)
