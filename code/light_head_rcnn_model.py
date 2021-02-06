from functools import partial
import dataset

from utils.py_faster_rcnn_utils.cython_nms import nms
from detection_opr.box_utils.box import DetBox
from detection_opr.utils.bbox_transform import clip_boxes, bbox_transform_inv
import network_desp
from config import cfg, config

import tensorflow as tf
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np

class LightHeadRCNN(object):
    def __init__(self, session):
        self.CAR_INDEX = 3
        self.session = session
        with self.session.as_default():
            # build net
            self.net = network_desp.Network()
            self.inputs = self.net.get_inputs()
            self.net.inference("TEST", self.inputs)
            self.saver = tf.train.Saver()

        test_collect_dict = self.net.get_test_collection()
        test_collect = list(test_collect_dict.values())
        self.val_func = partial(self.session.run, test_collect)

    def load(self, model_file):
        self.saver.restore(self.session, model_file)

    def get_detbox(self, image):
        ori_shape = image.shape

        if config.eval_resize == False:
            resized_img, scale = image, 1
        else:
            resized_img, scale = dataset.resize_img_by_short_and_max_size(
                image, config.eval_image_short_size, config.eval_image_max_size)
        height, width = resized_img.shape[0:2]

        resized_img = resized_img.astype(np.float32) - config.image_mean
        resized_img = np.ascontiguousarray(resized_img[:, :, [2, 1, 0]])

        im_info = np.array(
            [[height, width, scale, ori_shape[0], ori_shape[1], 0]],
            dtype=np.float32)

        feed_dict = {self.inputs[0]: resized_img[None, :, :, :], self.inputs[1]: im_info}

        _, scores, pred_boxes, rois = self.val_func(feed_dict=feed_dict)
        boxes = rois[:, 1:5] / scale

        if cfg.TEST.BBOX_REG:
            pred_boxes = bbox_transform_inv(boxes, pred_boxes)
            pred_boxes = clip_boxes(pred_boxes, ori_shape)

        pred_boxes = pred_boxes.reshape(-1, config.num_classes, 4)
        result_boxes = []
        # for j in range(1, config.num_classes):
        j = self.CAR_INDEX
        inds = np.where(scores[:, j] > config.test_cls_threshold)[0]
        cls_scores = scores[inds, j]
        cls_bboxes = pred_boxes[inds, j, :]
        cls_dets = np.hstack((cls_bboxes, cls_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(cls_dets, config.test_nms)
        cls_dets = np.array(cls_dets[keep, :], dtype=np.float, copy=False)
        for i in range(cls_dets.shape[0]):
            db = cls_dets[i, :]
            dbox = DetBox(
                db[0], db[1], db[2] - db[0], db[3] - db[1],
                tag=config.class_names[j], score=db[-1])
            result_boxes.append(dbox)
        if len(result_boxes) > config.test_max_boxes_per_image:
            result_boxes = sorted(
                result_boxes, reverse=True, key=lambda t_res: t_res.score) \
                [:config.test_max_boxes_per_image]
        # TODO: handle no detect?
        box = result_boxes[0]
        return box


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
