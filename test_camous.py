# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import cfg, config

import os
import random
import argparse
from collections import OrderedDict
from functools import partial

import yaml
import cv2
import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
import numpy as np

from utils.py_faster_rcnn_utils.cython_nms import nms

from detection_opr.box_utils.box import DetBox
from detection_opr.utils.bbox_transform import clip_boxes, bbox_transform_inv
# from detection_opr.utils import loss_opr_without_box_weight

import carla
from carla_sim import CarlaSimulator, _save_image
import dataset
import network_desp

_BATCH_NORM_EPSILON = 1e-5

class AvgMeter(object):
    def __init(self):
        self.reset()

    def reset(self):
        self.sum = 0.
        self.avg = 0.
        self.recent = 0.
        self.n = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.n += n
        self.avg = self.sum / self.n
        self.recent = value

class FasterRCNNModel(object):
    def __init__(self, session):
        self.CAR_INDEX = 3
        self.session = session
        with self.session.as_default():
            # build net
            self.net = network_desp.Network()
            self.inputs = self.net.get_inputs()
            self.net.inference('TEST', self.inputs)
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
    def __init__(self, session, camous_size=16, lr=0.1, momentum=0.9, l2_reg=0.1, back_shape=64, fore_shape=64,
                 use_bn=True, batch_norm_momentum=0.997):
        self.sess = session

        # TODO: lr need decay!!!
        # configs
        self.scope_name = "clone_network"
        self.camous_size = camous_size
        self.camous_branch_layers = int(np.log2(self.camous_size)) - 2
        self.l2_reg = l2_reg
        self.lr = lr
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

    def build_placeholders(self):
        self.camous_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.back_input = tf.placeholder(tf.float32, shape=[None, self.back_shape, self.back_shape, 3])
        self.fore_input = tf.placeholder(tf.float32, shape=[None, self.fore_shape, self.fore_shape, 3])
        self.loss_weight = tf.placeholder(tf.float32, shape=[None, 1])
        self.label = tf.placeholder(tf.float32, shape=[None, 1])
        self.batch_size_placeholder = tf.placeholder(tf.int32, shape=[1])
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
            self.batch_size_placeholder: [len(score_label)],
            self.loss_weight: [[1.]],
            self.training: False
        }

        return self.sess.run([self.camous_grads, self.loss_theta, self.loss_c], feed_dict=feed_dict)

    def accumulate_grads(self, camous_inputs, trans_inputs, score_label, loss_weight=1.):
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
            self.training: True
        }
        [loss_theta_v, loss_rec_v, loss_reg_v], _ = self.sess.run([[self.loss_theta, self.loss_rec, self.loss_reg], self.accum_ops], feed_dict=feed_dict)
        return loss_theta_v, loss_rec_v, loss_reg_v

    def step(self):
        """
        Step the clone network weights using the current accumulated gradients.
        """
        self.sess.run(self.train_step)
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
            merge = layers_lib.fully_connected(merge, 1, activation_fn=None)
            self.loss_rec = tf.losses.sigmoid_cross_entropy(label, merge, weights=self.loss_weight)
            self.loss_reg = tf.losses.get_regularization_loss(scope=self.scope_name)
            zero_label = tf.reshape(tf.tile(tf.zeros(label.shape[1], label.dtype), multiples=self.batch_size_placeholder),
                                    (-1, label.shape[1]))

            loss_c = tf.losses.sigmoid_cross_entropy(zero_label, merge)
            self.merge = merge
            self.loss_theta = self.loss_rec + self.loss_reg
            self.loss_c = tf.reduce_mean(loss_c)

            # optimizers, train steps, gradients
            self.camous_grads = tf.gradients(loss_c, inputs[0])
            self.global_step = tf.Variable(0, trainable=False)
            # FIXME: temporary, a restart lr decay might be prefered (use placeholder for lr)
            learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                       100, 0.9, staircase=True)
            self.optimizer_theta = tf.train.MomentumOptimizer(learning_rate, momentum=self.momentum)

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

class Attacker(object):
    def __init__(self, simulator, target_model, session, save_dir, display_info=True,
                 bounds=(0, 255),
                 camous_size=16, iters=50, num_init_camous=50, num_max_camous=400,
                 theta_max_num_ct_per_inneriter=1024, theta_random_ct=True, theta_random_ct_sample=True,
                 camous_std=5, num_add_camous=20,
                 batch_size=32, theta_optimize_iters=20, theta_early_stop_window=2, c_optimize_iters=10,
                 lr_c=4.0, pgd=True,
                 clone_net_cfg=None):
        # long-term TODO: restrict camous not to be too noisy
        self.target_model = target_model
        self.simulator = simulator
        self.session = session
        self.save_dir = save_dir
        self.display_info = display_info
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.log_file = open(os.path.join(save_dir, "attack.log"), "w")

        # configs
        self.bounds = bounds
        self.camous_size = camous_size
        self.iters = iters
        self.num_init_camous = num_init_camous
        self.num_max_camous = num_max_camous
        self.batch_size = batch_size
        self.theta_optimize_iters = theta_optimize_iters
        self.c_optimize_iters = c_optimize_iters
        self.theta_max_num_ct_per_inneriter = theta_max_num_ct_per_inneriter
        self.theta_random_ct = theta_random_ct
        self.theta_random_ct_sample = theta_random_ct_sample
        self.camous_std = camous_std
        self.num_add_camous = num_add_camous
        self.clone_net_cfg = clone_net_cfg or {}
        self.lr_c = lr_c
        self.pgd = pgd
        self.theta_early_stop_window = theta_early_stop_window

        # clone network
        self.clone_net = CloneNetwork(self.session, self.camous_size, **self.clone_net_cfg)
        self.camous_history = []
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

    def ct_data_gen(self):
        # generate camous, transform data pair randomly or sequentially
        transform_range = list(range(self._num_transforms))
        num_camous = len(self.camous_history)
        camous_range = list(range(num_camous))
        if self.theta_random_ct:
            random.shuffle(camous_range)
            random.shuffle(transform_range)

        num_gened = 0
        if self.theta_max_num_ct_per_inneriter is None:
            total_to_gen = self._num_transforms * num_camous
        else:
            total_to_gen = self.theta_max_num_ct_per_inneriter
        while num_gened < total_to_gen:
            if self.theta_random_ct_sample:
                i_camous = np.random.choice(camous_range)
                i_transform = np.random.choice(transform_range)
            else:
                # always finish all transforms of one camous before move on to another camous
                i_camous = num_gened / self._num_transforms
                i_transform = num_gened % self._num_transforms
            num_gened += 1
            yield i_camous, self.camous_history[i_camous], i_transform, self.transforms[i_transform], total_to_gen - num_gened
        return

    def get_transform_encoding(self, transform, image, detbox):
        # get background, foregorund
        foreground = image[int(detbox.y):int(detbox.y1), int(detbox.x):int(detbox.x1), :].copy()
        background = image.copy()
        background[int(detbox.y):int(detbox.y1), int(detbox.x):int(detbox.x1), :] = 0
        return background, foreground

    def print(self, *args, **kwargs):
        ret = print(*args, **kwargs)
        kwargs["file"] = self.log_file
        print(*args, **kwargs)
        self.log_file.flush()
        return ret

    def _get_display_info(self, detbox, loss_meters, addis=[]):
        return ["score: {:.3f}".format(detbox.score)] + \
            ["{}: {:.3f} (avg {:.3f})".format(n, m.recent, m.avg) for n, m in loss_meters.items()] + addis

    def run_attack(self):
        for _ in range(self.num_init_camous):
            # generate init camouflages
            self.camous_history.append((self.bounds[1] - self.bounds[0]) * \
                                       np.random.rand(self.camous_size, self.camous_size, 3) + \
                                       self.bounds[0])
            self.camous_weights = [1.] * len(self.camous_history)

        num_accum = 0
        for i_iter in range(self.iters):
            self.print("Alternate {}".format(i_iter))
            # ALTERNATION 1: tune clone network using current camous history
            num_updates = 0
            loss_v_meter = AvgMeter()
            loss_rec_meter = AvgMeter()
            loss_l2_meter = AvgMeter()
            camous_batch, transform_encodings, score_label_batch, loss_weight_batch = np.zeros(
                (0, self.camous_size, self.camous_size, 3)), [], np.zeros((0, 1)), np.zeros((0, 1))
            best_inner_i = 0
            best_inner_loss = np.inf
            for i_inner in range(self.theta_optimize_iters):
                loss_v_meter.reset()
                loss_rec_meter.reset()
                loss_l2_meter.reset()
                for i_camous, camous, i_trans, transform, remain in self.ct_data_gen():
                    # FIXME: the images can be cached instead of re-simulated using CARLA each time
                    image = self.simulator.get_image(camous, transform)
                    detbox = self.target_model.get_detbox(image)
                    if self.display_info:
                        # render the detbox and info onto the display
                        self.simulator.render_info(
                            [[detbox.x, detbox.y, detbox.w, detbox.h]] if detbox.score > 0.5 else [],
                            self._get_display_info(
                                detbox,
                                OrderedDict(zip(["loss_theta", "loss_rec", "loss_reg"],
                                                [loss_v_meter, loss_rec_meter, loss_l2_meter])),
                                addis=["Alter {} iter {} batch {}".format(i_iter, i_inner, num_updates), "camous {} trans {}".format(i_camous, i_trans)]))
                    # also render and print the clone network inputs (transform_encoding, camous) and the  loss infos (including running avg)
                    # all onto another pygame window.
                    if detbox.score < 0.5:
                        # mis-detection, might result in wrong foreground/background crop, might be misleading
                        continue
                    # FIXME: THIS is WRONG now!!! Now I use the image rendered using car with camous to crop out the
                    # "foreground" and "background". But, the original car and corresponding detbox should be used.
                    # As this is not related to the camous, only related to the camera transform (15 different transforms now),
                    # we can do this processing in advance -- Calculate out 15 transform encoding in advance.
                    transform_encoding = self.get_transform_encoding(transform, image, detbox)
                    transform_encodings.append(transform_encoding)
                    score_label_batch = np.concatenate([score_label_batch, [[detbox.score]]], axis=0)
                    camous_batch = np.concatenate([camous_batch, [camous]], axis=0)
                    loss_weight_batch = np.concatenate([loss_weight_batch, [[self.camous_weights[i_camous]]]], axis=0)
                    # TODO: move data gen to another thread, now gpu util is ver low...
                    num_accum += 1
                    if num_accum == self.batch_size:
                        assert len(camous_batch) == self.batch_size
                        loss_v, loss_rec, loss_l2 = self.clone_net.accumulate_grads(camous_batch, transform_encodings, score_label_batch,
                                                                              loss_weight=loss_weight_batch)
                        loss_v_meter.update(loss_v, self.batch_size)
                        loss_rec_meter.update(loss_rec, self.batch_size)
                        loss_l2_meter.update(loss_l2, self.batch_size)
                        self.clone_net.step()
                        num_accum = 0
                        num_updates += 1
                        camous_batch, transform_encodings, score_label_batch, loss_weight_batch = np.zeros((0, self.camous_size, self.camous_size, 3)), [], np.zeros((0, 1)), np.zeros((0, 1))
                        self.print("[Optimize Theta] Epoch/Batch {}/{}. loss: {:.3f} {:.3f} {:.3f}".format(
                            i_inner, num_updates, loss_v, loss_rec, loss_l2
                        ))
                if num_accum > 0:
                    loss_v, loss_rec, loss_l2 = self.clone_net.accumulate_grads(camous_batch, transform_encodings, score_label_batch,
                                                                                loss_weight=loss_weight_batch)
                    loss_v_meter.update(loss_v, num_accum)
                    loss_rec_meter.update(loss_rec, num_accum)
                    loss_l2_meter.update(loss_l2, num_accum)
                    self.clone_net.step()
                    num_accum = 0
                    num_updates += 1
                    self.print("[Optimize Theta] Epoch/Batch {}/{}. loss: {:.3f} {:.3f} {:.3f}".format(
                        i_inner, num_updates, loss_v, loss_rec, loss_l2
                    ))
                    camous_batch, transform_encodings, score_label_batch, loss_weight_batch = np.zeros((0, self.camous_size, self.camous_size, 3)), [], np.zeros((0, 1)), np.zeros((0, 1))
                self.print("[Optimize Theta] Epoch {} total batches {} (bs={}): mean loss: {:.3f} {:.3f} {:.3f}".format(
                    i_inner, num_updates, self.batch_size, loss_v_meter.avg, loss_rec_meter.avg, loss_l2_meter.avg))
                num_updates = 0
                if loss_v_meter.avg < best_inner_loss:
                    best_inner_i = i_inner
                    best_inner_loss = loss_v_meter.avg
                if i_inner - best_inner_i >= self.theta_early_stop_window:
                    print("mean loss does not decay for {} inner theta optimization iters, early-stop optimize theta".format(self.theta_early_stop_window))
                    break

            # ALTERNATION 2: attack using current clone network
            new_camous = self.camous_history[-1].copy()
            # camous_trace = [] # the camous update trace
            self.simulator.render_camous_trace(new_camous, index=0, clear=True)
            for i_inner in range(self.c_optimize_iters):
                transform_encodings, score_label_batch = [], np.zeros((0, 1))
                avg_score = 0.
                misdetect = 0
                for i_trans, transform in enumerate(self.transforms):
                    image = self.simulator.get_image(new_camous, transform)
                    detbox = self.target_model.get_detbox(image)
                    # record misdetection numbers
                    # TODO: maybe include ground truth box IoU is more solid
                    misdetect += detbox.score < 0.5
                    score_label_batch = np.concatenate([score_label_batch, [[detbox.score]]], axis=0)
                    avg_score += detbox.score
                    transform_encodings.append(self.get_transform_encoding(transform, image, detbox))
                    # render the detbox and info onto the display
                    if self.display_info:
                        self.simulator.render_info(
                            [[detbox.x, detbox.y, detbox.w, detbox.h]],
                            self._get_display_info(
                                detbox,
                                {},
                                addis=["Alter {} c iter {} trans {}".format(i_iter, i_inner, i_trans)]))

                grads, loss_theta_v, loss_c_v = self.clone_net.get_camous_grads(
                    np.tile(np.expand_dims(new_camous, 0), [self._num_transforms, 1, 1, 1]),
                    transform_encodings, score_label_batch)
                grads = np.mean(grads[0], axis=0)
                # TODO: attack maybe need momentum too..
                if self.pgd:
                    direct = np.sign(grads)
                else:
                    direct = grads
                avg_score /= self._num_transforms
                # update camous
                new_camous = new_camous + self.lr_c * direct
                # clip camous to acceptable color bounds
                new_camous.clip(min=self.bounds[0], max=self.bounds[1], out=new_camous)

                # print the diff
                diff = new_camous - self.camous_history[-1]
                linf_dist = np.max(np.abs(diff))
                l2_dist = np.linalg.norm(diff)
                self.print("[Optimize Camous] Iter {} (num eot transform={}) avg score: {}; mean loss theta: {}; mean loss c: {}; misdetect: {}/{}; diff: l2: {:.2f} linf: {:.2f}".format(
                    i_inner, self._num_transforms, avg_score, loss_theta_v, loss_c_v, misdetect, self._num_transforms, l2_dist, linf_dist))
                # display camous trace
                self.simulator.render_camous_trace(new_camous, index=i_inner + 1)

            # save the new camous
            save_path = os.path.join(self.save_dir, "camous_alter_{}.png".format(i_iter))
            self.print("Save the camous to {}".format(save_path))
            _save_image(new_camous, save_path)

            # save the current clone net
            save_path = os.path.join(self.save_dir, "clone_net_{}.ckpt".format(i_iter))
            self.print("Save the current clone net to {}".format(save_path))
            self.clone_net.save(save_path)

            # add random camous around the new camous, adjust camous weights
            # currently, use the same weights 1.0 for all the old camous
            self.camous_weights = [1.] * len(self.camous_history)
            for _ in range(self.num_add_camous):
                noise = self.camous_std * np.random.randn(self.camous_size, self.camous_size, 3)
                self.camous_history.append(np.clip(new_camous + noise, a_min=self.bounds[0], a_max=self.bounds[1]))
            self.camous_history.append(new_camous)
            self.camous_weights = self.camous_weights + [10.] * (self.num_add_camous + 1)

            if len(self.camous_history) > self.num_max_camous:
                self.camous_history = self.camous_history[-self.num_max_camous:]
                self.camous_weights = self.camous_weights[-self.num_max_camous:]

def main(device, model_file, sim_cfg, attacker_cfg, save_dir, display_info):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)

    detect_model = FasterRCNNModel(session)
    print("Constructed detect model")
    simulator = CarlaSimulator(**sim_cfg)
    print("Constructed simulator")
    attacker = Attacker(simulator, detect_model, session, save_dir, display_info, **attacker_cfg)
    print("Constructed attacker")

    session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    detect_model.load(model_file)
    print("Loaded detect model")

    print("Begin attack...")
    attacker.run_attack()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--display", default=":8")
    parser.add_argument("--device", default="0")
    parser.add_argument("--detect-model", default="model_dump/epoch_20.ckpt")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-display-info", action="store_true", default=False)
    args = parser.parse_args()
    os.environ["DISPLAY"] = args.display
    if args.seed is not None:
        seed = args.seed
        print("Setting random seed: {}.".format(seed))
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)

    with open(args.cfg_file) as cf:
        cfgs = yaml.safe_load(cf)

    main(args.device, args.detect_model, cfgs["simulator_cfg"], cfgs["attacker_cfg"],
         args.save_dir, not args.no_display_info)


