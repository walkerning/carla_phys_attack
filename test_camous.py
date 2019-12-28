# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg, config

import os
import sys
import random
import shutil
import logging
import cPickle
import argparse
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

import carla
from sim_pool import SimulatorPool
from carla_sim import _save_image
import dataset
import network_desp

import time

_BATCH_NORM_EPSILON = 1e-5

def _pickle_data(filename, data):
    with open(filename, "wb") as wf:
        cPickle.dump(data, wf)

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
                 num_init_camous=50, num_max_camous=400,
                 # how to produce camous transform in each theta epoch
                 theta_max_num_ct_per_innerepoch=1024, theta_random_ct_sample=True,
                 # newly added camous each alternation
                 camous_std=5, num_add_camous=40,
                 # theta optimization hyperparameters
                 batch_size=32, theta_optimize_epochs=20, theta_early_stop_window=2,
                 theta_early_stop_according_valid=False,
                 lr_theta_type="ExpDecay",
                 lr_theta_cfg={
                     "base_lr": 0.05,
                     "decay_alter_every": -1,
                     "decay_alter_rate": 1.0,
                     "decay_epoch_every": -1,
                     "decay_epoch_rate": 1.0
                 },
                 valid_ratio=0.1,
                 # camous optimization hyperparameters
                 c_optimize_iters=20, lr_c=2.0, pgd=True,
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
        self.valid_camous_history = []
        # camous weights
        self.camous_weights = []
        # transforms
        self.transforms = [
            carla.Transform(carla.Location(x=-4.5, z=1.5)),
            carla.Transform(carla.Location(x=5.0, z=1.7), carla.Rotation(yaw=180)),
            carla.Transform(carla.Location(x=4.0, z=1.7), carla.Rotation(yaw=180)),
            carla.Transform(carla.Location(y=4.0, z=1.5), carla.Rotation(yaw=-90)),
            carla.Transform(carla.Location(y=-4.0, z=1.5), carla.Rotation(yaw=90)),
            carla.Transform(carla.Location(y=3.0, x=3.0, z=1.5), carla.Rotation(yaw=-135)),
            carla.Transform(carla.Location(y=2.6, x=2.6, z=1.5), carla.Rotation(yaw=-135)),
            carla.Transform(carla.Location(y=5.0, x=5.0, z=1.0), carla.Rotation(yaw=-135)),
            carla.Transform(carla.Location(y=-3.0, x=3.0, z=1.5), carla.Rotation(yaw=135)),
            carla.Transform(carla.Location(y=-2.6, x=2.6, z=1.5), carla.Rotation(yaw=135)),
            carla.Transform(carla.Location(y=-5.0, x=5.0, z=1.0), carla.Rotation(yaw=135)),
            carla.Transform(carla.Location(y=-3.0, x=-3.0, z=1.5), carla.Rotation(yaw=45)),
            carla.Transform(carla.Location(y=-5.0, x=-5.0, z=1.0), carla.Rotation(yaw=45)),
            carla.Transform(carla.Location(y=3.0, x=-3.0, z=1.5), carla.Rotation(yaw=-45)),
            carla.Transform(carla.Location(y=5.0, x=-5.0, z=1.0), carla.Rotation(yaw=-45))
        ]

        self._num_transforms = len(self.transforms)
        self.simulator_pool = SimulatorPool(self.transforms, simulator_cfgs, batch_size=self.batch_size)

    def fix_detbox(self):
        self.i_trans2detbox = []
        self.ori_image_list = []
        self.transform_encodings = []
        self.clean_scores = []
        import ipdb
        ipdb.set_trace()
        self.clean_image_list = self.simulator_pool.get_images([(None, i) for i in range(self._num_transforms)])
        self.clean_detboxes = [self.target_model.get_detbox(image) for image in self.clean_image_list]
        self.clean_scores = [detbox.score for detbox in self.clean_detboxes]
        self.transform_encodings = [self.get_transform_encoding(transform, image, detbox)
                                    for transform, image, detbox in
                                    zip(self.transforms, self.clean_image_list, self.clean_detboxes)]
        for image, detbox in zip(self.clean_image_list, self.clean_detboxes):
            # plot box for manual check
            image_with_det = image.copy()
            image_with_det[int(detbox.y):int(detbox.y1), int(detbox.x), :] = [255, 0, 0]
            image_with_det[int(detbox.y):int(detbox.y1), int(detbox.x1), :] = [255, 0, 0]
            image_with_det[int(detbox.y), int(detbox.x):int(detbox.x1), :] = [255, 0, 0]
            image_with_det[int(detbox.y1), int(detbox.x):int(detbox.x1), :] = [255, 0, 0]
            _save_image(image, os.path.join(self.save_dir, "init_fixbox_{:d}.jpg".format(i)))
            _save_image(image_with_det,
                        os.path.join(self.save_dir, "init_fixbox_withdet_{:d}.jpg".format(i)))
            _save_image(self.transform_encodings[-1][0],
                        os.path.join(self.save_dir, "transform_encoding_{}_back.jpg".format(i)))
            _save_image(self.transform_encodings[-1][1],
                        os.path.join(self.save_dir, "transform_encoding_{}_fore.jpg".format(i)))
        self.logger.info("Clean detection scores for {} transforms: {}".format(len(self.transforms)), self.clean_scores)

    def get_epoch_cts(self, valid=False, total_to_gen=None, theta_random_ct_sample=None):
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
            return [(c_ind, camous_list[c_ind], t_ind) for c_ind, t_ind in zip(c_inds, t_inds)]
        else:
            camous_range = list(range(num_camous))
            transform_range = list(range(self._num_transforms))
            random.shuffle(camous_range)
            random.shuffle(transform_range)
            c_inds = camous_range[:(total_to_gen + self._num_transforms - 1) // self._num_transforms]
            return sum([[(c_ind, camous_list[c_ind], t_ind) for t_ind in transform_range]
                        for c_ind in c_inds], [])[:total_to_gen]

    def get_transform_encoding(self, transform, image, detbox):
        # get background, foregorund
        foreground = image[int(detbox.y):int(detbox.y1), int(detbox.x):int(detbox.x1), :].copy()
        background = image.copy()
        background[int(detbox.y):int(detbox.y1), int(detbox.x):int(detbox.x1), :] = 0
        return background, foreground

    def _get_display_info(self, detbox, loss_meters, addis=[]):
        return ["score: {:.3f}".format(detbox.score)] + \
            ["{}: {:.3f} (avg {:.3f})".format(n, m.recent, m.avg) for n, m in loss_meters.items()] + addis

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

            scores = np.array([self.target_model.get_detbox(image).score for image in images])
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


    def run_attack(self, load_clone_net=None):
        self.fix_detbox()

        if load_clone_net is not None:
            self.clone_net.load(load_clone_net)

        for _ in range(self.num_init_camous):
            # generate init camouflages
            self.camous_history.append((self.bounds[1] - self.bounds[0]) * \
                                       np.random.rand(self.camous_size, self.camous_size, 3) + \
                                       self.bounds[0])
        self.camous_weights = [1.] * len(self.camous_history)
        if self._num_init_valid > 0:
            for _ in range(self._num_init_valid):
                self.valid_camous_history.append((self.bounds[1] - self.bounds[0]) * \
                                                 np.random.rand(self.camous_size, self.camous_size, 3) + \
                                                 self.bounds[0])
            self.valid_camous_weights = [1.] * len(self.valid_camous_history)
        else:
            self.valid_camous_weights = []
        print("hi")

        for i_iter in range(self.iters):
            self.logger.info("Alternate {}".format(i_iter))
            if i_iter == 0 and load_clone_net is not None:
                # if load clone network from disk, skip the clone net training in the first alternation
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

                # recordings for early stop theta in this alternation
                best_inner_i = 0
                best_inner_loss = np.inf

                start_time = time.time()
                print("hi1")
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
                    epoch_cts = self.get_epoch_cts()
                    self.simulator_pool.put_cts([[c, t_ind] for _, c, t_ind in epoch_cts])
                    if self.valid_camous_history:
                        valid_epoch_cts = self.get_epoch_cts(
                            valid=True, total_to_gen=len(self.valid_camous_history) * self._num_transforms,
                            theta_random_ct_sample=False)
                        self.simulator_pool.put_cts([[c, t_ind] for _, c, t_ind in valid_epoch_cts], new_epoch=True)
                    cur_num = 0
                    num_batches = len(self.simulator_pool)
                    # begin epoch
                    print("hi2")
                    for i_batch, batch in enumerate(self.simulator_pool): # batched camera-captured 2-d images rendered by the simulators
                        num_data = len(batch)
                        end_num = cur_num + num_data
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
                        print("hi3")

                        if (i_batch + 1) % self.print_every == 0:
                            self.logger.info(("Alter {}/Epoch {}/Batch {} "
                                        "lr: {:.3f}. loss: {:.3f} {:.3f} {:.3f}").format(
                                i_iter, i_epoch, i_batch, cur_lr, loss_v, loss_rec, loss_l2))
                        cur_num = end_num
                    self.logger.info(("[Train Theta] Alter {}/Epoch {} total batches {} (bs={}): "
                                "mean loss: {:.3f} {:.3f} {:.3f}; "
                                "min det score using current camous set:"
                                " {:.3f} (mean {:.3f})").format(
                                    i_iter, i_epoch, num_batches, self.batch_size,
                                    loss_v_meter.avg, loss_rec_meter.avg, loss_l2_meter.avg,
                                    det_score_meter.min, det_score_meter.avg))
                    # end epoch

                    # valid each epoch
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
                            valid_loss_v_meter.update(loss_v, num_data)
                            valid_loss_rec_meter.update(loss_rec, num_data)
                            valid_loss_l2_meter.update(loss_l2, num_data)
                            valid_det_score_meter.update(score_label_batch.mean(), num_data)
                        self.logger.info(("[Valid Theta] Alter {}/Epoch {} total valid batches {} (bs={}): "
                                          "mean loss: {:.3f} {:.3f} {:.3f}; "
                                          "min det score using current camous set:"
                                          " {:.3f} (mean {:.3f})").format(
                                              i_iter, i_epoch, num_valid_batches, self.batch_size,
                                              valid_loss_v_meter.avg, valid_loss_rec_meter.avg, valid_loss_l2_meter.avg,
                                              valid_det_score_meter.min, valid_det_score_meter.avg))
                    # end valid
                    # check early stop according to valid loss or train loss
                    check_v_meter = self.valid_loss_v_meter if self.theta_early_stop_according_valid else self.loss_v_meter
                    if check_v_meter.avg < best_inner_loss:
                        best_inner_i = i_epoch
                        best_inner_loss = check_v_meter.avg
                    if i_epoch - best_inner_i >= self.theta_early_stop_window:
                        self.logger.info("mean loss does not decay for {} inner theta optimization iters,"
                                         " early-stop optimize theta".format(self.theta_early_stop_window))
                        break
                self.logger.info("time for theta:", time.time() - start_time)

                # save the current clone net
                save_path = os.path.join(self.save_dir, "clone_net_{}.ckpt".format(i_iter))
                self.logger.info("Save the current clone net to {}".format(save_path))
                self.clone_net.save(save_path)

            # ---- ALTERNATION 2: attack using current clone network ----
            new_camous = self.camous_history[-1].copy()
            # self.simulator.render_camous_trace(new_camous, index=0, clear=True)
            init_loss_theta_v = None
            for i_inner in range(self.c_optimize_iters):
                # get the camera-captured image with camous from different camera perspective
                images = self.simulator_pool.get_images(
                    [(new_camous, t_ind) for t_ind in range(self._num_transforms)])
                detboxes = [self.target_model.get_detbox(image) for image in images]
                scores = np.array([detbox.score for detbox in detboxes])[:, None]
                # record misdetection number and avg score
                misdetect = (scores < 0.5).sum()
                avg_score = scores.mean()
                # TODO: check according to ground truth box IoU is more solid

                # TODO: render all the 15 detection results onto pygame display
                grads, loss_theta_v, loss_c_v = self.clone_net.get_camous_grads(
                    np.tile(np.expand_dims(new_camous, 0), [self._num_transforms, 1, 1, 1]),
                    self.transform_encodings, scores
                )
                if init_loss_theta_v is None:
                    init_loss_theta_v = loss_theta_v
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
                # clip camous to acceptable color bounds
                new_camous.clip(min=self.bounds[0], max=self.bounds[1], out=new_camous)

                # print the diff
                diff = new_camous - self.camous_history[-1]
                linf_dist = np.max(np.abs(diff))
                l2_dist = np.linalg.norm(diff)
                self.logger.info(("[Optimize Camous] Iter {} (num eot transform={}) avg score: {}; "
                            "mean loss theta: {}; mean loss c: {}; misdetect: {}/{}; "
                            "DIFF: l2: {:.2f} linf: {:.2f}").format(
                                i_inner, self._num_transforms, avg_score, loss_theta_v, loss_c_v,
                                misdetect, self._num_transforms, l2_dist, linf_dist))
                # TODO: display camous trace
                # self.simulator.render_camous_trace(new_camous, index=i_inner + 1)

            # save the new camous
            save_path = os.path.join(self.save_dir, "camous_alter_{}.png".format(i_iter))
            self.logger.info("Save the camous to {}".format(save_path))
            _save_image(new_camous, save_path)

            # add random camous around the new camous, adjust camous weights
            # currently, use the same weights 1.0 for all the old camous
            self.camous_weights = [1.] * len(self.camous_history)
            self.camous_history += list(np.clip(new_camous + self.camous_std * np.random.randn(
                self._num_add_camous, self.camous_size, self.camous_size, 3),
                                                a_min=self.bounds[0], a_max=self.bounds[1]))
            self.camous_weights = self.camous_weights + [5.] * self.num_add_camous

            # add valid camous
            self.valid_camous_weights = [1.] * len(self.valid_camous_history)
            self.valid_camous_history += list(np.clip(new_camous + self.camous_std * np.random.randn(
                self._num_add_valid_camous, self.camous_size, self.camous_size, 3),
                                                      a_min=self.bounds[0], a_max=self.bounds[1]))
            self.valid_camous_weights = self.valid_camous_weights + [5.] * self._num_add_valid_camous

            if len(self.camous_history) > self.num_max_camous:
                self.camous_history = self.camous_history[-self.num_max_camous:]
                self.camous_weights = self.camous_weights[-self.num_max_camous:]

            if len(self.valid_camous_history) > self._num_max_valid_camous:
                self.valid_camous_history = self.valid_camous_history[-self.num_max_camous:]
                self.valid_camous_weights = self.valid_camous_weights[-self.num_max_camous:]

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
    parser.add_argument("--device", default="0")
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
