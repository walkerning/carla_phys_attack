# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from colorprint import *

from config import cfg, config

import os
import sys
import random
import shutil
import logging
import cPickle
import argparse
# from collections import OrderedDict
from functools import partial

import yaml
import cv2
import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
import numpy as np
# python2 only have skimage 0.14.x
# from skimage.draw import rectangle_perimeter

from utils.py_faster_rcnn_utils.cython_nms import nms

from detection_opr.box_utils.box import DetBox
from detection_opr.utils.bbox_transform import clip_boxes, bbox_transform_inv
# from detection_opr.utils import loss_opr_without_box_weight

import carla
from sim_pool import SimulatorPool
from carla_sim import _save_image, _magnify_camous
import dataset
import network_desp

import time

_BATCH_NORM_EPSILON = 1e-5

def _pickle_data(filename, data):
    with open(filename, "wb") as wf:
        cPickle.dump(data, wf)

def _load_data(filename):
    with open(filename, "rb") as rf:
        data = cPickle.load(rf)
    return data

def _pickle_transforms(filename, transforms):
    to_pickle = [[(transform.location.x, transform.location.y, transform.location.z),
                  (transform.rotation.pitch, transform.rotation.roll, transform.rotation.yaw)]
                 for transform in transforms]
    with open(filename, "wb") as wf:
        cPickle.dump(to_pickle, wf)

def _load_pickle_transforms(filename):
    with open(filename, "rb") as rf:
        ts = cPickle.load(rf)
    transforms = [carla.Transform(
        carla.Location(**dict(zip(["x", "y", "z"], t[0]))),
        carla.Rotation(**dict(zip(["pitch", "roll", "yaw"], t[1])))) for t in ts]
    return transforms

def _mkdir_or_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

class AvgMeter(object):
    def __init(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.avg = 0.
        self.recent = 0.
        self.n = 0
        self.min = None
        self.max = None

    def update(self, value, n=1):
        self.sum += value * n
        self.n += n
        self.avg = self.sum / self.n
        self.recent = value
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value < self.max:
            self.max = value

class FasterRCNNModel(object):
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

class CloneNetwork(object):
    def __init__(self, session, camous_size=16, momentum=0.9, l2_reg=0.001, back_shape=64, fore_shape=64,
                 use_bn=True, batch_norm_momentum=0.997):
        self.sess = session

        # configs
        self.scope_name = "clone_network"
        self.camous_size = camous_size
        self.camous_branch_layers = int(np.log2(self.camous_size)) - 2
        self.l2_reg = l2_reg
        self.momentum = momentum
        self.back_shape = back_shape
        self.fore_shape = fore_shape
        self.use_bn = use_bn
        self.batch_norm_momentum = batch_norm_momentum

        with session.as_default():
            self.build_placeholders()
            self.build([self.camous_input, self.back_input, self.fore_input], self.label)
        self.saver = tf.train.Saver(tf.global_variables(scope=self.scope_name))

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)

    def build_placeholders(self):
        self.lr = tf.placeholder(tf.float32, shape=[])
        self.camous_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.back_input = tf.placeholder(tf.float32, shape=[None, self.back_shape, self.back_shape, 3])
        self.fore_input = tf.placeholder(tf.float32, shape=[None, self.fore_shape, self.fore_shape, 3])
        self.loss_weight = tf.placeholder(tf.float32, shape=[None, 1])
        self.label = tf.placeholder(tf.float32, shape=[None, 1])
        self.training = tf.placeholder(tf.bool, shape=[])

    def _handle_trans_inputs(self, trans_inputs):
        assert len(trans_inputs) == 2, "This clone network accept a 2-ele tuple as transform encoding"
        # cv2.resize will auto consider the first k-dim and leave the else unchanged.
        # return: (self.back_shape, self.back_shape, 3), (self.fore_shape, self.fore_shape, 3)
        return cv2.resize(trans_inputs[0], (self.back_shape, self.back_shape), interpolation=cv2.INTER_LINEAR), \
            cv2.resize(trans_inputs[1], (self.fore_shape, self.fore_shape), interpolation=cv2.INTER_LINEAR)

    def get_camous_grads(self, camous_inputs, trans_inputs, score_label):
        if len(camous_inputs.shape) == 3:
            trans_inputs = self._handle_trans_inputs(trans_inputs)
            camous_inputs = [camous_inputs]
            back_inputs = [trans_inputs[0]]
            fore_inputs = [trans_inputs[1]]
            score_label = [score_label]
        else:
            # batched
            trans_inputs = [self._handle_trans_inputs(trans_input) for trans_input in trans_inputs]
            back_inputs = np.stack([trans_input[0] for trans_input in trans_inputs])
            fore_inputs = np.stack([trans_input[1] for trans_input in trans_inputs])
        feed_dict = {
            self.camous_input: camous_inputs,
            self.back_input: back_inputs,
            self.fore_input: fore_inputs,
            self.label: score_label,
            self.loss_weight: [[1.]],
            self.training: False
        }

        return self.sess.run([self.camous_grads, self.loss_theta, self.loss_c], feed_dict=feed_dict)

    def accumulate_grads(self, camous_inputs, trans_inputs, score_label, loss_weight=1., acc_grad=True):
        if len(camous_inputs.shape) == 3:
            trans_inputs = self._handle_trans_inputs(trans_inputs)
            camous_inputs = [camous_inputs]
            back_inputs = [trans_inputs[0]]
            fore_inputs = [trans_inputs[1]]
            score_label = [score_label]
        else:
            # batched
            trans_inputs = [self._handle_trans_inputs(trans_input) for trans_input in trans_inputs]
            back_inputs = np.stack([trans_input[0] for trans_input in trans_inputs])
            fore_inputs = np.stack([trans_input[1] for trans_input in trans_inputs])
        # print("back_inputs", np.shape(back_inputs))
        feed_dict = {
            self.camous_input: camous_inputs,
            self.back_input: back_inputs,
            self.fore_input: fore_inputs,
            self.label: score_label,
            self.loss_weight: loss_weight,
            self.training: acc_grad
        }
        if acc_grad:
            [loss_theta_v, loss_rec_v, loss_reg_v], _ = self.sess.run([[self.loss_theta, self.loss_rec, self.loss_reg], self.accum_ops], feed_dict=feed_dict)
        else:
            loss_theta_v, loss_rec_v, loss_reg_v = self.sess.run([self.loss_theta, self.loss_rec, self.loss_reg], feed_dict=feed_dict)
        return loss_theta_v, loss_rec_v, loss_reg_v

    def step(self, lr):
        """
        Step the clone network weights using the current accumulated gradients.
        """
        self.sess.run(self.train_step, feed_dict={self.lr: lr})
        self.sess.run(self.zero_agrad_op)

    def batch_norm(self, inputs, data_format="channel_last"):
        if self.use_bn:
            return tf.layers.batch_normalization(
                inputs=inputs, axis=1 if data_format == "channels_first" else 3,
                momentum=self.batch_norm_momentum, epsilon=_BATCH_NORM_EPSILON, center=True,
                scale=True, training=self.training, fused=True)
        else:
            return inputs

    def build(self, inputs, label):
        # TODO: the structure of clone network needs modification
        #  not rational to resize to 64 * 64
        # camous branch
        x = inputs[0]
        with tf.variable_scope(self.scope_name):
            for i in range(self.camous_branch_layers):
                x = layers_lib.conv2d(x, 4, 3, weights_regularizer=tf.keras.regularizers.l2(l=self.l2_reg), activation_fn=None)
                x = self.batch_norm(x)
                x = tf.nn.relu(x)
                x = layers_lib.max_pool2d(x, 2)
            # background branch
            y = inputs[1]
            for i in range(4):
                y = layers_lib.conv2d(y, 4, 3, weights_regularizer=tf.keras.regularizers.l2(l=self.l2_reg), activation_fn=None)
                y = self.batch_norm(y)
                y = tf.nn.relu(y)
                y = layers_lib.max_pool2d(y, 2)
            # foreground branch
            z = inputs[2]
            for i in range(4):
                z = layers_lib.conv2d(z, 4, 3, weights_regularizer=tf.keras.regularizers.l2(l=self.l2_reg), activation_fn=None)
                z = self.batch_norm(z)
                z = tf.nn.relu(z)
                z = layers_lib.max_pool2d(z, 2)
            # merge
            merge = tf.concat([x, y, z], axis=3)
            merge = layers_lib.conv2d(merge, 4, 3, weights_regularizer=tf.keras.regularizers.l2(l=self.l2_reg), activation_fn=None)
            merge = self.batch_norm(merge)
            merge = tf.nn.relu(merge)
            merge = layers_lib.flatten(merge)
            # merge = layers_lib.fully_connected(merge, 1, activation_fn=tf.sigmoid)
            merge = layers_lib.fully_connected(merge, 1, activation_fn=None)

            # TODO: pay attention to below
            # the difference between actual detection score and the clone network predicted score
            # EPS = 1e-5
            # label very small (-> 0) or very large (->1), merge must be ->0 and ->1
            # self.loss_rec = tf.reduce_mean(- merge * tf.log(label + EPS) - (1 - merge) * tf.log(1 - label + EPS))
            # merge very small (-> 0) or very large (->1), label must be ->0 and ->1
            # self.loss_rec = tf.reduce_mean(- label * tf.log(merge + EPS) - (1 - label) * tf.log(1 - merge + EPS))
            self.loss_rec = tf.losses.sigmoid_cross_entropy(label, merge, weights=self.loss_weight)
            self.loss_reg = tf.losses.get_regularization_loss(scope=self.scope_name)

            zero_label = tf.zeros([tf.shape(x)[0], 1], tf.float32)
            loss_c = tf.losses.sigmoid_cross_entropy(zero_label, merge)
            # loss_c = tf.reduce_mean(- tf.log(1 - merge + EPS)) # very strange, grad=0

            self.merge = merge
            self.loss_theta = self.loss_rec + self.loss_reg
            self.loss_c = loss_c

            # optimizers, train steps, gradients
            self.camous_grads = tf.gradients(loss_c, inputs[0])
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer_theta = tf.train.MomentumOptimizer(self.lr, momentum=self.momentum)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope_name)
            tvs = tf.trainable_variables(scope=self.scope_name)
            accum_vars = [tf.Variable(tf.zeros_like(tv), trainable=False) for tv in tvs]
            self.zero_agrad_op = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
            self.grads_and_vars = self.optimizer_theta.compute_gradients(self.loss_theta, tvs)
            # NOTE: the batch norm update is done every small iter (hope it will not cause severe vibration)
            with tf.control_dependencies(update_ops):
                self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(self.grads_and_vars)]
            
            self.train_step = self.optimizer_theta.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(self.grads_and_vars)],
                                                                   global_step=self.global_step)


class ExpDecayLrScheduler(object):
    def __init__(self, base_lr, decay_alter_every=-1, decay_alter_rate=1.0, decay_epoch_every=-1, decay_epoch_rate=1.0):
        self.base_lr = base_lr
        self.decay_alter_every = decay_alter_every
        self.decay_alter_rate = decay_alter_rate
        self.decay_epoch_every = decay_epoch_every
        self.decay_epoch_rate = decay_epoch_rate

    def get_lr(self, alter, epoch):
        if self.decay_alter_every > 0:
            alter_base_lr = self.base_lr * self.decay_alter_rate ** (alter // self.decay_alter_every)
        else:
            alter_base_lr = self.base_lr
        if self.decay_epoch_every > 0:
            lr = alter_base_lr * self.decay_epoch_rate ** (epoch // self.decay_epoch_every)
        else:
            lr = alter_base_lr
        return lr


class Attacker(object):
    def __init__(self, simulator_cfgs, target_model, session, save_dir, display_info=False,
                 bounds=(0, 255),
                 camous_size=16,
                 # alternation num
                 iters=50,
                 # camous number
                 num_init_camous=50, num_max_camous=500,
                 # how to produce camous transform in each theta epoch
                 theta_max_num_ct_per_innerepoch=1600, theta_random_ct_sample=True,
                 # newly added camous each alternation
                 camous_std=5, num_add_camous=40,
                 # number of new camous to be attacked during alternation 2
                 new_camous_to_attack=5,
                 # number of attack results to be randomly expanded
                 new_camous_attack_results=1,
                 # theta optimization hyperparameters
                 batch_size=32, theta_optimize_epochs=5, theta_early_stop_window=2,
                 theta_early_stop_according_valid=False,
                 lr_theta_type="ExpDecay",
                 lr_theta_cfg={
                     "base_lr": 0.05,
                     "decay_alter_every": -1,
                     "decay_alter_rate": 1.0,
                     "decay_epoch_every": -1,
                     "decay_epoch_rate": 1.0
                 },
                 valid_ratio=0.4,
                 # camous optimization hyperparameters
                 c_optimize_iters=20, lr_c=3.0, pgd=True,
                 clone_net_cfg=None,
                 # other
                 print_every=1):
        # long-term TODO: restrict camous not to be too noisy
        self.target_model = target_model
        self.session = session
        self.save_dir = save_dir

        self.display_info = display_info
        self.logger = logging.getLogger("Attacker")
        # configs
        self.bounds = bounds
        self.camous_size = camous_size
        self.iters = iters
        self.num_init_camous = num_init_camous
        self.num_max_camous = num_max_camous

        self.new_camous_to_attack = new_camous_to_attack
        self.new_camous_attack_results = new_camous_attack_results

        self.batch_size = batch_size
        self.theta_optimize_epochs = theta_optimize_epochs
        self.c_optimize_iters = c_optimize_iters
        self.theta_max_num_ct_per_innerepoch = theta_max_num_ct_per_innerepoch
        self.theta_random_ct_sample = theta_random_ct_sample
        self.camous_std = camous_std
        self.num_add_camous = num_add_camous
        self.clone_net_cfg = clone_net_cfg or {}
        self.lr_theta_scheduler = globals()[lr_theta_type + "LrScheduler"](**lr_theta_cfg)
        self.valid_ratio = valid_ratio
        self._num_add_valid_camous = int((self.num_add_camous - 1) * self.valid_ratio)
        self._num_init_valid = int(self.num_init_camous * self.valid_ratio)
        self._num_max_valid_camous = int(self.num_max_camous * self.valid_ratio)
        self.lr_c = lr_c
        self.pgd = pgd
        self.theta_early_stop_window = theta_early_stop_window
        self.theta_early_stop_according_valid = theta_early_stop_according_valid
        self.print_every = print_every

        # clone network
        self.clone_net = CloneNetwork(self.session, self.camous_size, **self.clone_net_cfg)
        self.camous_history = []
        self.camous_index_history = []
        self.valid_camous_history = []

        self.camous_score_history = []
        self.valid_camous_score_history = []

        self.solid_history = []
        self.solid_camous_history = []
        self.solid_camous_score_history = []
        self.solid_camous_remark_history = []
        # camous weights
        self.camous_weights = []
        # transforms
        # TODO: modify  transfotms
        #  test set can be more refined than train set
        #  test and draw the sensitiveness towards attacking of different positions
        #  some transforms have only changed distance, and the detbox cutoff similar,not making sense
        z_axis = [1.2] # [1.0, 1.2, 1.4, 1.6]
        distances = [6.0] # [5.0, 8.0, 12.0, 16.0]
        angles = 180 - np.linspace(0, 360, 25)
        self.transforms_lib = []
        for d in distances:
            for z in z_axis:
                for a in angles:
                    x = -np.cos(a*3.1415926/180) * d
                    y = -np.sin(a*3.1415926/180) * d
                    self.transforms_lib.append(carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=a)))
        self.transforms = self.transforms_lib
        self._num_transforms = len(self.transforms)
        self.simulator_pool = SimulatorPool(self.transforms, simulator_cfgs, batch_size=self.batch_size)

    def one_stream(self, camous, save_dir='./save_eval', save_encodings=False):
        self.transform_encodings = []
        self.scores = []
        self.image_list = self.simulator_pool.get_images([(camous, i) for i in range(self._num_transforms)])
        self.detboxes = [self.target_model.get_detbox(image) for image in self.image_list]
        self.scores = [detbox.score for detbox in self.detboxes]
        if save_encodings:
            self.transform_encodings = [self.get_transform_encoding(transform, image, detbox)
                                        for transform, image, detbox in
                                        zip(self.transforms, self.image_list, self.detboxes)]
        for i, (image, detbox) in enumerate(zip(self.image_list, self.detboxes)):
            # plot box for manual check
            image_with_det = image.copy()
            image_with_det[int(detbox.y):int(detbox.y1), int(detbox.x), :] = [255, 0, 0]
            image_with_det[int(detbox.y):int(detbox.y1), int(detbox.x1), :] = [255, 0, 0]
            image_with_det[int(detbox.y), int(detbox.x):int(detbox.x1), :] = [255, 0, 0]
            image_with_det[int(detbox.y1), int(detbox.x):int(detbox.x1), :] = [255, 0, 0]

            image_with_det = cv2.putText(image_with_det, 'score:{:.3f}'.format(detbox.score), (int(detbox.x) - 3, int(detbox.y)),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
            tmp_save_dir = os.path.join(self.save_dir, save_dir)
            _mkdir_or_exists(tmp_save_dir)
            # _save_image(image, os.path.join(tmp_save_dir, "init_fixbox_{:d}.jpg".format(i)))
            _save_image(image_with_det, os.path.join(tmp_save_dir, "Image_withdet_{:d}.jpg".format(i)))
            if save_encodings:
                _save_image(self.transform_encodings[i][0],
                            os.path.join(tmp_save_dir, "transform_encoding_{}_back.jpg".format(i)))
                _save_image(self.transform_encodings[i][1],
                            os.path.join(tmp_save_dir, "transform_encoding_{}_fore.jpg".format(i)))
        self.logger.info("Detection scores for {} transforms: {}".format(len(self.transforms), self.scores))

    def get_epoch_cts(self, valid=False, total_to_gen=None, theta_random_ct_sample=None, shuffle=False):
        camous_list = self.valid_camous_history if valid else self.camous_history
        num_camous = len(camous_list)
        if theta_random_ct_sample is None:
            theta_random_ct_sample = self.theta_random_ct_sample
        if total_to_gen is None:
            if self.theta_max_num_ct_per_innerepoch is None:
                total_to_gen = self._num_transforms * num_camous
            else:
                total_to_gen = self.theta_max_num_ct_per_innerepoch
        if theta_random_ct_sample:
            c_inds = np.random.randint(low=0, high=num_camous, size=total_to_gen)
            t_inds = np.random.randint(low=0, high=self._num_transforms, size=total_to_gen)
            data_type = 'Training' if not valid else 'Validation'
            print('[{} Data] Generating {:d} randomly selected ct-pairs, each camou from range {:d} and each transfrom from range {:d} '.format(
                data_type, total_to_gen, num_camous, self._num_transforms), color='green')
            return [(c_ind, camous_list[c_ind], t_ind) for c_ind, t_ind in zip(c_inds, t_inds)]
        else:
            camous_range = list(range(num_camous))
            transform_range = list(range(self._num_transforms))
            if shuffle:
                random.shuffle(camous_range)
                random.shuffle(transform_range)
            c_gen = (total_to_gen + self._num_transforms - 1) // self._num_transforms
            c_inds = camous_range[:c_gen]
            print('Generating {:d} ct-pairs, with {:d} camous and {:d} transforms for each camou'.format(
                total_to_gen, c_gen, self._num_transforms), color='green')
            whole_set = sum([[(c_ind, camous_list[c_ind], t_ind) for t_ind in transform_range]
                        for c_ind in c_inds], [])[:total_to_gen]
            # shuffle the whole set, even when validation
            if shuffle:
                random.shuffle(whole_set)
            return whole_set

    def get_transform_encoding(self, transform, image, detbox):
        # get background, foregorund
        foreground = image[int(detbox.y):int(detbox.y1), int(detbox.x):int(detbox.x1), :].copy()
        background = image.copy()
        background[int(detbox.y):int(detbox.y1), int(detbox.x):int(detbox.x1), :] = 0
        # print('image.shape',np.shape(image))
        # print('forground.shape', np.shape(foreground))
        return background, foreground


    def _get_display_info(self, detbox, loss_meters, addis=[]):
        return ["score: {:.3f}".format(detbox.score)] + \
            ["{}: {:.3f} (avg {:.3f})".format(n, m.recent, m.avg) for n, m in loss_meters.items()] + addis

    def run_attack(self, load_dir=None):

        self.one_stream(camous=None, save_dir='detbox_init', save_encodings=True)
        # load clone net and camous history
        if load_dir is not None:
            self._load_data_and_network(load_dir)
        else:
            for _ in range(self.num_init_camous):
                # generate init camouflages
                self.camous_history.append((self.bounds[1] - self.bounds[0]) * \
                                           np.random.rand(self.camous_size, self.camous_size, 3) + \
                                           self.bounds[0])
            if self._num_init_valid > 0:
                for _ in range(self._num_init_valid):
                    self.valid_camous_history.append((self.bounds[1] - self.bounds[0]) * \
                                                     np.random.rand(self.camous_size, self.camous_size, 3) + \
                                                     self.bounds[0])
                self.valid_camous_weights = [1.] * len(self.valid_camous_history)
            else:
                self.valid_camous_weights = []

        self.camous_weights = [1.] * len(self.camous_history)
        self.valid_camous_weights = [1.] * len(self.valid_camous_history)

        for i_iter in range(self.iters):
            self.logger.info("Alternate {}".format(i_iter))
            stop_epoch = None
            if i_iter == 0 and load_dir is not None:
                # if load clone network from disk, skip the clone net training in the first alternation
                self.logger.info("Loading clone network from  disk and  skip the fisrt alternation")
                pass
            else:
                # ---- ALTERNATION 1: tune clone network using current camous history ----
                loss_v_meter = AvgMeter()
                loss_rec_meter = AvgMeter()
                loss_l2_meter = AvgMeter()
                det_score_meter = AvgMeter()
                # valid meters
                valid_loss_v_meter = AvgMeter()
                valid_loss_rec_meter = AvgMeter()
                valid_loss_l2_meter = AvgMeter()
                valid_det_score_meter = AvgMeter()

                # reset all the scores for each camou in history
                self.camous_score_history = []
                self.valid_camous_score_history = []
                self.tmp_ct = [[] for _ in range(len(self.camous_history))]
                self.transform_scores = []
                for i in range(len(self.camous_history)):
                    self.camous_score_history.append(AvgMeter())
                    self.camous_score_history[i].reset()
                for i in range(len(self.valid_camous_history)):
                    self.valid_camous_score_history.append(AvgMeter())
                    self.valid_camous_score_history[i].reset()
                for i in range(self._num_transforms):
                    self.transform_scores.append(AvgMeter())
                    self.transform_scores[i].reset()
                # recordings for early stop theta in this alternation
                best_inner_i = 0
                best_inner_loss = np.inf

                start_time = time.time()
                for i_epoch in range(self.theta_optimize_epochs):
                    cur_lr = self.lr_theta_scheduler.get_lr(i_iter, i_epoch)
                    # reset meters
                    loss_v_meter.reset()
                    loss_rec_meter.reset()
                    loss_l2_meter.reset()
                    det_score_meter.reset()
                    valid_loss_v_meter.reset()
                    valid_loss_rec_meter.reset()
                    valid_loss_l2_meter.reset()
                    valid_det_score_meter.reset()
                    # put data requests into simulator pool
                    epoch_cts = self.get_epoch_cts(theta_random_ct_sample=False)
                    self.simulator_pool.put_cts([[c, t_ind] for _, c, t_ind in epoch_cts])
                    if self.valid_camous_history:
                        valid_epoch_cts = self.get_epoch_cts(
                            valid=True, total_to_gen=len(self.valid_camous_history) * self._num_transforms,
                            theta_random_ct_sample=False)
                        self.simulator_pool.put_cts([[c, t_ind] for _, c, t_ind in valid_epoch_cts], new_epoch=True)
                    cur_num = 0
                    init_len = len(self.simulator_pool)
                    num_batches = len(self.simulator_pool)
                    # begin epoch
                    for i_batch, batch in enumerate(self.simulator_pool):
                        # batched camera-captured 2-d images rendered by the simulators
                        num_data = len(batch)
                        end_num = cur_num + num_data
                        assert num_data, print(num_data)
                        assert end_num <= len(epoch_cts), print(len(epoch_cts), end_num)
                        c_inds, t_inds = zip(*[(item[0], item[2]) for item in epoch_cts[cur_num:end_num]])
                        transform_encodings = np.stack([self.transform_encodings[i_trans] for i_trans in t_inds])
                        loss_weight_batch = np.array([self.camous_weights[i_camous] for i_camous in c_inds])[:, None]
                        detboxes = [self.target_model.get_detbox(image) for image in batch]
                        score_label_batch = np.array([detbox.score for detbox in detboxes])[:, None]
                        camous_batch = np.stack([epoch_cts[_num][1] for _num in range(cur_num, end_num)])
                        # run update
                        loss_v, loss_rec, loss_l2 = self.clone_net.accumulate_grads(
                            camous_batch, transform_encodings, score_label_batch,
                            loss_weight=loss_weight_batch)
                        self.clone_net.step(lr=cur_lr)
                        loss_v_meter.update(loss_v, num_data)
                        loss_rec_meter.update(loss_rec, num_data)
                        loss_l2_meter.update(loss_l2, num_data)
                        det_score_meter.update(score_label_batch.mean(), num_data)

                        if (i_batch + 1) % self.print_every == 0:
                            self.logger.info(("Alter {}/Epoch {}/Batch {} "
                                        "lr: {:.3f}. loss:  v {:.3f} | rec {:.3f} | l2 {:.3f}; ").format(
                                i_iter, i_epoch, i_batch, cur_lr, loss_v, loss_rec, loss_l2))
                        cur_num = end_num

                        # update camous scores at each epoch, because we cannot figure out the end epoch due to early stop
                        # score_label_batch = [[0.999], [0.999], ...]
                                   
                        check_c_inds = [1, 3, 5]
                        check_t_inds = [0,1,2]

                        for c_ind, t_ind, score, image in zip(c_inds, t_inds, score_label_batch, batch):
                            self.camous_score_history[c_ind].update(score[0])
                            self.transform_scores[t_ind].update(score[0])
                            # self.tmp_ct[c_ind].append(t_ind)
                            # # check  synchronism
                            if c_ind in check_c_inds and t_ind in check_t_inds:
                                tmp_save_dir = os.path.join(self.save_dir, "check_tmp_e{}".format(i_epoch))
                                _magnify_camous(self.camous_history[c_ind], save_name=os.path.join(tmp_save_dir, "camou_big_c{:d}.jpg".format(c_ind)), save_size=2048)
                                _mkdir_or_exists(tmp_save_dir)
                                _save_image(image, os.path.join(tmp_save_dir, "image_c{:d}_t{:d}_score{:.4f}.jpg".format(c_ind, t_ind, score[0])))
                                _save_image(self.camous_history[c_ind], os.path.join(tmp_save_dir, "camou_c{:d}.jpg".format(c_ind)))

                        if i_batch >= init_len - 1:
                            break

                    self.logger.info(("[Train Theta] Alter {}/Epoch {} total batches {} (bs={}): "
                                "mean loss: v {:.3f} | rec {:.3f} | l2 {:.3f}; \n"
                                "For all BATCHES, min det score using current camous set:"
                                " {:.3f}, mean {:.3f}").format(
                                    i_iter, i_epoch, num_batches, self.batch_size,
                                    loss_v_meter.avg, loss_rec_meter.avg, loss_l2_meter.avg,
                                    det_score_meter.min, det_score_meter.avg))

                    # end epoch
                    # TODO: Strange! if for each epoch, all the transfroms for all camous go through, the det.score.mean shouldn't varies a lot
                    # new_camous = self.camous_process_in_theta(save_num=3, valid=False, outer_iter=i_iter, inner_iter=i_epoch)

                    # validate each epoch
                    if self.valid_camous_history:
                        cur_num = 0
                        num_valid_batches = len(self.simulator_pool)
                        for i_batch, batch in enumerate(self.simulator_pool): # batched camera-captured 2-d images rendered by the simulators
                            num_data = len(batch)
                            end_num = cur_num + num_data
                            c_inds, t_inds = zip(*[(item[0], item[2]) for item in valid_epoch_cts[cur_num:end_num]])
                            transform_encodings = np.stack([self.transform_encodings[i_trans] for i_trans in t_inds])
                            loss_weight_batch = np.array([self.valid_camous_weights[i_camous] for i_camous in c_inds])[:, None]
                            detboxes = [self.target_model.get_detbox(image) for image in batch]
                            score_label_batch = np.array([detbox.score for detbox in detboxes])[:, None]
                            camous_batch = np.stack([valid_epoch_cts[_num][1] for _num in range(cur_num, end_num)])

                            # run update
                            loss_v, loss_rec, loss_l2 = self.clone_net.accumulate_grads(
                                camous_batch, transform_encodings, score_label_batch,
                                loss_weight=loss_weight_batch, acc_grad=False)
                            # while not tepping?

                            valid_loss_v_meter.update(loss_v, num_data)
                            valid_loss_rec_meter.update(loss_rec, num_data)
                            valid_loss_l2_meter.update(loss_l2, num_data)
                            valid_det_score_meter.update(score_label_batch.mean(), num_data)
                        self.logger.info(("[Valid Theta] Alter {}/Epoch {} total valid batches {} (bs={}): "
                                          "mean loss: {:.3f} {:.3f} {:.3f}; \n"
                                          "For all BATCHES, min det score using current camous set:"
                                          " {:.3f}, mean {:.3f}").format(
                                              i_iter, i_epoch, num_valid_batches, self.batch_size,
                                              valid_loss_v_meter.avg, valid_loss_rec_meter.avg, valid_loss_l2_meter.avg,
                                              valid_det_score_meter.min, valid_det_score_meter.avg))
                    # end valid
                    # check early stop according to valid loss or train loss
                    check_v_meter = valid_loss_v_meter if self.theta_early_stop_according_valid else loss_v_meter
                    if check_v_meter.avg < best_inner_loss:
                        best_inner_i = i_epoch
                        best_inner_loss = check_v_meter.avg
                    if i_epoch - best_inner_i >= self.theta_early_stop_window:
                        self.logger.info("mean loss does not decay for {} inner theta optimization iters,"
                                         " early-stop optimize theta".format(self.theta_early_stop_window))
                        stop_epoch = i_epoch
                        break
                self.logger.info("time for theta:", time.time() - start_time)
                # save the current clone net and camous
                self._save_data_and_network(output_dir=os.path.join(self.save_dir, 'checkpoint_iter-{:d}'.format(i_iter)))
            if stop_epoch is None:
                stop_epoch = self.theta_optimize_epochs


            # ---- ALTERNATION 2: attack using current clone network ----
            new_camous_batch, new_scores_batch = self.camous_process_in_theta(save_num=self.new_camous_to_attack,
                                                                              valid=False, outer_iter=i_iter, inner_iter=stop_epoch)
            # self.simulator.render_camous_trace(new_camous, index=0, clear=True)
            for i_camous, new_camous in enumerate(new_camous_batch):
                init_loss_theta_v = None
                init_avg_score = None
                init_camous = new_camous.copy()
                attack_trace = []
                self.logger.info("Attacking [{:d}/{:d}] camou, init score: {}".format(i_camous+1, len(new_camous_batch), new_scores_batch[i_camous]))
                for i_inner in range(self.c_optimize_iters):
                    # get the camera-captured image with camous from different camera perspective
                    images = self.simulator_pool.get_images(
                        [(new_camous, t_ind) for t_ind in range(self._num_transforms)])
                    detboxes = [self.target_model.get_detbox(image) for image in images]
                    scores = np.array([detbox.score for detbox in detboxes])[:, None]
                    # record misdetection number and avg score
                    misdetect = (scores < 0.5).sum()
                    avg_score = scores.mean()

                    # Fixme: using clone network grad
                    '''  using clone network grad 
                    # TODO: check according to ground truth box IoU is more solid
                    grads, loss_theta_v, loss_c_v = self.clone_net.get_camous_grads(
                        np.tile(np.expand_dims(new_camous, 0), [self._num_transforms, 1, 1, 1]),
                        self.transform_encodings, scores
                    )
                    if init_loss_theta_v is None:
                        init_loss_theta_v = loss_theta_v
                        init_avg_score = avg_score
                    if loss_theta_v > 1.5 * init_loss_theta_v:
                        # Stop to follow the clone network gradient, since the camous input already
                        # enter the region that the clone network cannot fit the Simulator-Detector blackbox well
                        self.logger.info("Stop to follow the clone network gradient, since the camous input already "
                                   "enter the region that the clone network cannot fit the Simulator-Detector blackbox well. "
                                   "now fit(theta) loss: {:.3f} (init {:.3f})".format(loss_theta_v, init_loss_theta_v))
                        break
                    grads = np.mean(grads[0], axis=0)
                    # TODO: attack maybe need momentum and 2-order momentum of gradients
                    if self.pgd:
                        direct = np.sign(grads)
                    else:
                        direct = grads
                    # update camous
                    new_camous = new_camous - self.lr_c * direct
                    '''
                    loss_theta_v = loss_c_v = 0
                    direct = np.random.rand(self.camous_size, self.camous_size, 3)
                    new_camous = new_camous + direct * self.lr_c
                    # clip camous to acceptable color bounds

                    new_camous.clip(min=self.bounds[0], max=self.bounds[1], out=new_camous)

                    # print the diff
                    diff = new_camous - init_camous
                    linf_dist = np.max(np.abs(diff))
                    l2_dist = np.linalg.norm(diff)

                    self.logger.info(("[Optimize Camous] Iter {} (num eot transform={}) avg score: {}; "
                                "mean loss theta: {:.3f}; mean loss c: {:.3f}; misdetect: {}/{}; "
                                "DIFF: l2: {:.2f} linf: {:.2f}").format(
                                    i_inner, self._num_transforms, avg_score, loss_theta_v, loss_c_v,
                                    misdetect, self._num_transforms, l2_dist, linf_dist))

                    attack_trace.append([new_camous.copy(), avg_score.copy(), loss_theta_v, loss_c_v, misdetect, i_inner])

                inner_camous = [ att[0] for att in attack_trace]
                inner_scores = [ att[1] for att in attack_trace]
                    # self.simulator.render_camous_trace(new_camous, index=i_inner + 1)

                # select the best new camous and update camous_history
                attack_trace = sorted(attack_trace, key=lambda x: x[1])
                for i in range(self.new_camous_attack_results):
                    new_camous, avg_score, loss_theta_v, loss_c_v, misdetect, i_inner = attack_trace[i]
                    self.logger.info("Best camous [{:d}/{:d}] for No.{} to attack at iter:{:d}, "
                                     "score from {:.3f} to {:.3f}, loss theta:{:.3f}, loss c:{:.3f}, misdetect:{}/{}".format(
                        i, self.new_camous_attack_results, i_camous, i_inner, init_avg_score, avg_score, loss_theta_v, loss_c_v, misdetect, self._num_transforms
                    ))

                    self.update_solid_camous([new_camous], [avg_score], ['Alter-{}_opt-camou_{}'.format(i_iter, i)])

                    # or just use the solid ones?
                    clear_all = True if i == 0 and i_camous == 0 else False
                    self.update_camous_history(new_camous=new_camous, avg_score=avg_score, init_camous=init_camous,
                                               save_tag="Alter-{}_C-{}_No-{}".format(i_iter, i_camous, i), clear_all=clear_all)

    def camous_process_in_theta(self, save_num=1, valid=False, outer_iter=0, inner_iter=0):
        """ init new camous from camous history to run alternation2
        :param save_num:
        :return: new camous batch as initiation for attacker """
        avg_trans_scores = [trans.avg for trans in self.transform_scores]
        print(avg_trans_scores)
        raw_score_list = self.camous_score_history if not valid else self.valid_camous_score_history
        camou_list = self.camous_history if not valid else self.valid_camous_history
        # select the lowest but not zero scores
        score_list = [score.avg if score.n > 0 else 100.0 for score in raw_score_list]
        # for c_ind in range(len(self.camous_history)):
        #     print(len(self.tmp_ct[c_ind]))
        #     print(self.tmp_ct[c_ind])
        # for score in raw_score_list:
        #     print(score.n)
        #     assert score.n % self._num_transforms == 0, print('Not 15 n!')
        rank = np.argsort(score_list) # from the lowest to highest
        camous_batch = []
        score_batch = []
        remark_batch = []
        # TODO: find the ones with lowest theta-loss, and try to exclude the case where the selected camous are with few transforms
        #  --- Done
        log_text = '[Camous in Theta Opt] Adding {:d} new camous to solid history:'.format(save_num)
        for i in range(save_num):
            camous_batch.append(camou_list[rank[i]].copy())
            score_batch.append(score_list[rank[i]])
            remark_batch.append('Alter-{}_Epoch-{}_Index_Opt-theta'.format(outer_iter, inner_iter))
            log_text += '  Index {}, score: {:.4f}'.format(rank[i], score_list[rank[i]])
            # if save_dir is not None:
            #     _save_image(camou_list[rank[i]], save_dir + 'camou_{:.3f}.jpg'.format(score_list[rank[i]]))
        print()
        self.logger.info(log_text)
        # save all the solid camous
        # camous_batch, score_batch = self.update_solid_camous(camous_batch, score_batch, remark_batch, extract_num=self.new_camous_to_attack)
        return camous_batch, score_batch

    def _save_data_and_network(self, output_dir=None):
        self.logger.info('Saving data')
        output_dir = os.path.join(self.save_dir, 'checkpoints') if output_dir is None else output_dir
        _mkdir_or_exists(output_dir)
        _pickle_data(os.path.join(output_dir, "camous_history.pkl"), self.camous_history)
        _pickle_data(os.path.join(output_dir, "valid_camous_history.pkl"), self.valid_camous_history)
        _pickle_transforms(os.path.join(output_dir, "transforms.pkl"), self.transforms)
        self.logger.info('Saving all the data to direction [{}], in {}, {}, and {}'.format(
            output_dir, "camous_history.pkl", "valid_camous_history.pkl", "transforms.pkl"
        ))
        save_path = os.path.join(output_dir, "clone_net.ckpt")
        self.logger.info("Saving the current clone net to {}".format(save_path))
        self.clone_net.save(save_path)
        return

    def _load_data_and_network(self, load_dir):
        assert load_dir is not None, print('Load_dir cannot be None! ')

        load_clone_net = os.path.join(load_dir, "clone_net.ckpt")
        self.clone_net.load(load_clone_net)
        self.logger.info("Loading clone net from {}".format(load_clone_net))

        self.camous_history = _load_data(os.path.join(load_dir, "camous_history.pkl"))
        self.valid_camous_history = _load_data(os.path.join(load_dir, "valid_camous_history.pkl"))
        # self.transforms = _load_pickle_transforms(os.path.join(load_dir, "transforms.pkl"))
        self.logger.info('Loading all the data from direction [{}], in {}, {}'.format(
            load_dir, "camous_history.pkl", "valid_camous_history.pkl"
        ))
        # TODO: currently the transforms are fixed and not loaded
        return

    def update_camous_history(self, new_camous, avg_score, init_camous=None, save_tag='tag', clear_all=False):
        """ Updating training and validating set
        add random camous around the new camous, adjust camous weights
        actually it's important on how to add the random steps
        :param new_camous_batch:
        :return: """
        if clear_all:
            self.camous_history = []
            self.valid_camous_history = []
        # save the new camous
        camous_dir = os.path.join(self.save_dir, "attacked_camous")
        _mkdir_or_exists(camous_dir)
        save_path = os.path.join(camous_dir, save_tag + 'camou.jpg')
        self.logger.info("Save the camous to {}".format(save_path))
        _save_image(new_camous, save_path)

        if init_camous is not None:
            saliency_mask = new_camous - init_camous
            save_path = os.path.join(camous_dir, save_tag + 'diff.jpg')
            self.logger.info("Save the diff_saliency mask to {}".format(save_path))
            _save_image(saliency_mask, save_path)
        else:
            saliency_mask = np.ones(new_camous.shape)

        self.camous_weights = [1.] * len(self.camous_history)
        self.camous_weights = self.camous_weights + [1.] * self.num_add_camous

        rand_steps = np.random.rand(self.num_add_camous - 1, self.camous_size, self.camous_size, 3) # rand means all pos
        rand_new_camous = init_camous + self.camous_std * saliency_mask * rand_steps
        self.camous_history += list(np.clip(rand_new_camous, a_min=self.bounds[0], a_max=self.bounds[1]))
        self._expand_index_history(self.num_add_camous)

        # self.camous_history += list(np.clip(new_camous + self.camous_std * np.random.randn(
        #     self.num_add_camous, self.camous_size, self.camous_size, 3),
        #                                     a_min=self.bounds[0], a_max=self.bounds[1]))

        # add valid camous
        self.valid_camous_weights = [1.] * len(self.valid_camous_history)
        self.valid_camous_history += list(np.clip(new_camous + self.camous_std * np.random.randn(
            self._num_add_valid_camous, self.camous_size, self.camous_size, 3),
                                                  a_min=self.bounds[0], a_max=self.bounds[1]))
        self.valid_camous_weights = self.valid_camous_weights + [1.] * self._num_add_valid_camous

        if len(self.camous_history) > self.num_max_camous:
            self.camous_history = self.camous_history[-self.num_max_camous:]
            self.camous_weights = self.camous_weights[-self.num_max_camous:]

        if len(self.valid_camous_history) > self._num_max_valid_camous:
            self.valid_camous_history = self.valid_camous_history[-self.num_max_camous:]
            self.valid_camous_weights = self.valid_camous_weights[-self.num_max_camous:]
        pass

    def update_solid_camous(self, camous_list, scores_list, remark_list, extract_num=5):
        for camou, score, remark in zip(camous_list, scores_list, remark_list):
            self.solid_history.append([camou.copy(), score, remark])
        # sorted, low score in the front
        self.solid_history = sorted(self.solid_history, key=lambda x: x[1])
        extract_num = min(len(self.solid_history), extract_num)
        if extract_num > 0:
            camous_batch = []
            scores_batch = []
            log_text = '[Extract Solid Camous] Extracting {:d} camous with lowest detect scores: '.format(extract_num)
            for i in range(extract_num):
                log_text += '\n Remark:{} , score:{:.4f}'.format(self.solid_history[i][2], self.solid_history[i][1])
                camous_batch.append(self.solid_history[i][0].copy())
                scores_batch.append(self.solid_history[i][1])
            self.logger.info(log_text)
            return camous_batch, scores_batch
        return

    def _expand_index_history(self, number):
        return
        start = self.camous_index_history[-1]
        for i in range(number):
            self.camous_index_history.append(start + i + 1)
        assert len(self.camous_index_history) == len(self.camous_history)


    def gen_data(self, output_dir):
        # generate init camouflages
        self.camous_history = list(
            (self.bounds[1] - self.bounds[0]) * \
            np.random.rand(self.num_init_camous, self.camous_size, self.camous_size, 3) + \
            self.bounds[0])
        self.valid_camous_history = list(
            (self.bounds[1] - self.bounds[0]) * \
            np.random.rand(self._num_init_valid, self.camous_size, self.camous_size, 3) + \
            self.bounds[0])
        # get all the cts
        cts = self.get_epoch_cts(total_to_gen=self.num_init_camous * self._num_transforms,
                                 theta_random_ct_sample=False)
        self.simulator_pool.put_cts([[c, t_ind] for _, c, t_ind in cts])

        # valid cts
        if self.valid_camous_history:
            valid_cts = self.get_epoch_cts(total_to_gen=self._num_init_valid * self._num_transforms,
                                           theta_random_ct_sample=False)
            self.simulator_pool.put_cts([[c, t_ind] for _, c, t_ind in valid_cts], new_epoch=True)

        # dump train batches
        BATCH_PER_FILE = 40
        num_batches = len(self.simulator_pool)
        iter_ = iter(self.simulator_pool)
        parsed_batches = 0
        parsed_num = 0
        _pickle_data(os.path.join(output_dir, "camous_history.pkl"), self.camous_history)
        _pickle_transforms(os.path.join(output_dir, "transforms.pkl"), self.transforms)
        cs, ts = zip(*[(c_ind, t_ind) for c_ind, _, t_ind in cts])
        file_id = 0
        while 1:
            start_time = time.time()
            self.logger.info("Start generating data {}".format(file_id))
            file_name = os.path.join(output_dir, "data_batch_{}.pkl".format(file_id))
            num_b_file = min(BATCH_PER_FILE, num_batches - parsed_batches)
            images = np.concatenate([next(iter_) for _ in range(num_b_file)], axis=0)

            try:
                scores = np.array([self.target_model.get_detbox(image).score for image in images])
            except RuntimeError:
                print('Get box failed!')
                exit()
            num_datum = len(images)

            elapsed_time = time.time() - start_time
            self.logger.info("Elpased {:.1f} s. Saving {} data to {}".format(elapsed_time, num_datum, file_name))
            _pickle_data(file_name, {
                "camous_ids": cs[parsed_num:parsed_num+num_datum],
                "trans_ids": ts[parsed_num:parsed_num+num_datum],
                "images": images,
                "scores": scores
            })
            parsed_num += len(images)
            parsed_batches += num_b_file
            if parsed_batches >= num_batches:
                break
            file_id += 1

        # dump valid batch
        self.logger.info("Saving valid batches ...")
        file_name = os.path.join(output_dir, "valid_batch.pkl")
        valid_cs, valid_ts = zip(*[(c_ind, t_ind) for c_ind, _, t_ind in valid_cts])
        valid_images = np.concatenate([batch for batch in self.simulator_pool], axis=0)
        valid_scores = np.array([self.target_model.get_detbox(image).score for image in images])
        _pickle_data(file_name, {
                "camous_ids": valid_cs,
                "trans_ids": valid_ts,
                "images": valid_images,
                "scores": valid_scores
        })
        self.logger.info("Saving {} valid data to {}".format(len(valid_images), file_name))

        # camous list, transform list
        # per data: camous_id, transform_id, rendered image, score
        # return self.camous_history, self.transforms, cs, ts, images, scores

def main(device, model_file, sim_cfgs, attacker_cfg, save_dir, display_info, load_clone_net, gen_data_only):
    assert sim_cfgs
    # config tf
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)

    detect_model = FasterRCNNModel(session)
    logging.info("Constructed detect model")
    attacker = Attacker(sim_cfgs, detect_model, session, save_dir, display_info, **attacker_cfg)
    logging.info("Constructed simulator and attacker")

    session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    detect_model.load(model_file)
    logging.info("Loaded detect model")

    if not gen_data_only:
        logging.info("Begin attack...")
        attacker.run_attack(load_clone_net)
    else:
        # TODO: For now, only generate randomly
        # maybe save and load the previous camous history, and tessellate around those points
        attacker.gen_data(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--display", default=":8")
    parser.add_argument("--device", default="2")
    parser.add_argument("--detect-model", default="model_dump/epoch_20.ckpt")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--display-info", action="store_true", default=False)
    parser.add_argument("--load-clone-net", default=None)
    parser.add_argument("--gen-data-only", action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    shutil.copyfile(args.cfg_file, os.path.join(args.save_dir, "config.yaml"))

    # config logging
    LOG_FORMAT = "%(asctime)s %(name)-10s %(levelname)7s: %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=LOG_FORMAT, datefmt="%m/%d %I:%M:%S %p")
    log_file = os.path.join(args.save_dir, "attack.log")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)

    os.environ["DISPLAY"] = args.display
    if args.seed is not None:
        seed = args.seed
        logging.info("Setting random seed: {}.".format(seed))
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)

    with open(args.cfg_file) as cf:
        cfgs = yaml.safe_load(cf)
    main(args.device, args.detect_model, cfgs["simulator_cfg"], cfgs["attacker_cfg"],
         args.save_dir, args.display_info, args.load_clone_net, args.gen_data_only)
