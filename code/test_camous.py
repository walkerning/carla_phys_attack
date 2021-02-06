# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from colorprint import *
# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
#
# except IndexError:
#     pass

import os
import sys
import random
import shutil
import logging
import cPickle
import argparse
sys.path.append('/home/foxfi/unreal/carla_dynamic/PythonAPI/carla/dist')
# from collections import OrderedDict
from functools import partial
import yaml
import cv2
import tensorflow as tf
# from tensorflow.contrib import layers as layers_lib
import numpy as np
# python2 only have skimage 0.14.x
# from skimage.draw import rectangle_perimeter

import carla
from sim_pool import SimulatorPool
from carla_sim import _save_image, _magnify_camous

import time
import mmcv

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
        os.makedirs(path)


class AvgMeter(object):
    def __init(self):
        self.reset()

    def reset(self, avg=0.):
        self.sum = 0.
        self.avg = avg
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
    def __init__(self, simulator_cfgs, target_model, session, save_dir, display_info=False, train=True,
                 bounds=(0, 255),
                 camous_size=16,
                 # alternation num
                 iters=50,
                 # camous number
                 num_init_camous=100, num_max_camous=640, solid_max=10,
                 # how to produce camous transform in each theta epoch
                 theta_max_num_ct_per_innerepoch=12800, theta_random_ct_sample=True,
                 # newly added camous each alternation
                 camous_std=10, num_add_camous=  16,
                 # number of new camous to be attacked during alternation 2
                 new_camous_to_attack=3,
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
                 c_optimize_iters=20, lr_c=5.0, pgd=True,
                 clone_net_cfg=None,
                 # other
                 print_every=1):

        self.train = train
        self.test_random = True

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
        self.solid_max = solid_max

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
        # self.clone_net = CloneNetwork(self.session, self.camous_size, **self.clone_net_cfg)
        self.camous_history = []
        self.camous_index_history = []
        self.valid_camous_history = []

        self.camous_score_history = []
        self.valid_camous_score_history = []

        self.solid_history = []
        # camous weights
        self.camous_weights = []
        # transforms
        # TODO: modify  transfotms
        #  test set can be more refined than train set
        #  test and draw the sensitiveness towards attacking of different positions
        #  some transforms have only changed distance, and the detbox cutoff similar,not making sense
        if self.train:
            heights = [1.2, 1.2]
            distances = [5.0, 8.0]
            angles = 180 - np.linspace(0, 360, 9)
        else:
            heights =  [1.2, 1.2, 1.5, 1.5]
            distances = [5.0, 8.0, 12.0, 15.0]
            angles = 180 - np.linspace(0, 360, 25)
        self.transforms = []
        self.transforms_info = []
        for d, z in zip(distances, heights):
            for a in angles:
                x = -np.cos(a*np.pi/180) * d
                y = -np.sin(a*np.pi/180) * d
                self.transforms.append(carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=a)))
                self.transforms_info.append([x, y, z, a])
        self._num_transforms = len(self.transforms)
        self.simulator_pool = SimulatorPool(self.transforms, simulator_cfgs, batch_size=self.batch_size)

    def one_stream(self, camous, save_dir='./save_eval', save_encodings=False):
        self.transform_encodings = []
        image_list = self.simulator_pool.get_images([(camous, i) for i in range(self._num_transforms)])
        print('here 2')
        detboxes = [self.target_model.get_detbox(image) for image in image_list]
        scores = [detbox.score for detbox in detboxes]
        if save_encodings:
            self.transform_encodings = [self.get_transform_encoding(transform, image, detbox)
                                        for transform, image, detbox in
                                        zip(self.transforms, image_list, detboxes)]
        start = time.time()
        tmp_save_dir = os.path.join(self.save_dir, save_dir)
        _mkdir_or_exists(tmp_save_dir)
        for i, (image, detbox) in enumerate(zip(image_list, detboxes)):
            # plot box for manual check
            print(i)
            # todo: change to a more pleasant visualize function as in faster rcnn
            image_with_det = image.copy()
            image_with_det[int(detbox.y):int(detbox.y1), int(detbox.x), :] = [255, 0, 0]
            image_with_det[int(detbox.y):int(detbox.y1), int(detbox.x1), :] = [255, 0, 0]
            image_with_det[int(detbox.y), int(detbox.x):int(detbox.x1), :] = [255, 0, 0]
            image_with_det[int(detbox.y1), int(detbox.x):int(detbox.x1), :] = [255, 0, 0]
            font_size = image.shape[0]/256. * 0.5
            image_with_det = cv2.putText(image_with_det, 'score:{:.4f}'.format(detbox.score), (int(detbox.x) - 3, int(detbox.y)),
                                         cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,0,0),1)
            _save_image(image, os.path.join(tmp_save_dir, "Image_{:d}.jpg".format(i)))
            _save_image(image_with_det, os.path.join(tmp_save_dir, "Image_withdet_{:d}.jpg".format(i)))
            if save_encodings:
                _save_image(self.transform_encodings[i][0],
                            os.path.join(tmp_save_dir, "transform_encoding_{}_back.jpg".format(i)))
                _save_image(self.transform_encodings[i][1],
                            os.path.join(tmp_save_dir, "transform_encoding_{}_fore.jpg".format(i)))
        mmcv.dump(dict(scores=scores,transforms=self.transforms_info), os.path.join(tmp_save_dir, "score_location.pkl"))
        avg_score = np.mean(scores)
        precise = np.sum(np.asarray(scores)>0.5)/self._num_transforms
        self.logger.info("[Evaluating]{} transforms, time used:{:.2f}s AVG score: {:.4f} P0.5: {:.2f}".format(
            self._num_transforms, time.time()-start, avg_score, precise))
        return avg_score, precise

    def get_epoch_cts(self, valid=False, total_to_gen=None, theta_random_ct_sample=False, shuffle=False):
        camous_list = self.valid_camous_history if valid else self.camous_history
        num_camous = len(camous_list)
        num_transforms = self._num_transforms
        # if theta_random_ct_sample is None:
        #     theta_random_ct_sample = self.theta_random_ct_sample
        if total_to_gen is None:
            total_to_gen = num_transforms * num_camous
            # if self.theta_max_num_ct_per_innerepoch is None:
            #     total_to_gen = self._num_transforms * num_camous
            # else:
            #     total_to_gen = min(self.theta_max_num_ct_per_innerepoch, self._num_transforms * num_camous)
        if theta_random_ct_sample:
            c_inds = np.random.randint(low=0, high=num_camous, size=total_to_gen)
            t_inds = np.random.randint(low=0, high=self._num_transforms, size=total_to_gen)
            data_type = 'Training' if not valid else 'Validation'
            print('[{} Data] Generating {:d} randomly selected ct-pairs, each camou from range {:d} and each transfrom from range {:d} '.format(
                data_type, total_to_gen, num_camous, self._num_transforms), color='green')
            return [(c_ind, camous_list[c_ind], t_ind) for c_ind, t_ind in zip(c_inds, t_inds)]
        else:
            camous_range = list(range(num_camous))
            transform_range = random.sample(range(self._num_transforms), num_transforms)
            if shuffle:
                random.shuffle(camous_range)
                random.shuffle(transform_range)
            c_gen = (total_to_gen + num_transforms - 1) // num_transforms
            c_inds = camous_range[:c_gen]
            print('Generating {:d} ct-pairs, with {:d} camous and {:d} transforms for each camou. Wanted {} x {} pairs'.format(
                total_to_gen, c_gen, num_transforms, num_camous, num_transforms), color='green')
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
        return ["score: {:.4f}".format(detbox.score)] + \
            ["{}: {:.4f} (avg {:.4f})".format(n, m.recent, m.avg) for n, m in loss_meters.items()] + addis

    def run_attack(self, load_dir=None, test_camous=None): # './data/square_xuanyi.jpg'
        self.one_stream(camous=None, save_dir='detbox_init', save_encodings=True)
        # load clone net and camous history
        skip=False
        load_iter=0
        if load_dir is not None:
            skip, load_iter = self._load_data_and_network(load_dir)
            if not self.train:
                self.eval_solid_camous()
                if self.test_random:
                    self.camous_history = []
                    for _ in range(self.num_init_camous):
                        self.camous_history.append((self.bounds[1] - self.bounds[0]) * \
                                                   np.random.rand(self.camous_size, self.camous_size, 3) + \
                                                   self.bounds[0])
                else:
                    exit()
        else:
            for _ in range(self.num_init_camous):
                # generate init camouflages
                self.camous_history.append((self.bounds[1] - self.bounds[0]) * \
                                           np.random.rand(self.camous_size, self.camous_size, 3) + \
                                           self.bounds[0])
        self.camous_weights = [1.] * len(self.camous_history)
        self.valid_camous_weights = [1.] * len(self.valid_camous_history)

        if test_camous is not None:
            test_camous = cv2.resize(cv2.imread(test_camous), (self.camous_size, self.camous_size))
            self.one_stream(camous=test_camous, save_dir='test_camous', save_encodings=False)
            exit()

        for i_iter in range(load_iter, self.iters):
            self.logger.info(" * * * * * Alternate {} * * * * * ".format(i_iter))
            if i_iter == load_iter and skip:
                new_camous_batch, new_scores_batch = self.update_solid_camous([], [], [], extract_num=self.new_camous_to_attack)
            else:
                # ---- ALTERNATION 1: tune clone network using current camous history ----
                # reset all the scores for each camou in history
                self.camous_score_history = []
                self.tmp_ct = [[] for _ in range(len(self.camous_history))]
                self.transform_scores = []
                for i in range(len(self.camous_history)):
                    self.camous_score_history.append(AvgMeter())
                    self.camous_score_history[i].reset(1.)
                for i in range(self._num_transforms):
                    self.transform_scores.append(AvgMeter())
                    self.transform_scores[i].reset(1.)
                detected = np.zeros(len(self.camous_history))
                # recordings for early stop theta in this alternation

                start_time = time.time()
                # put data requests into simulator pool
                epoch_cts = self.get_epoch_cts(theta_random_ct_sample=False)
                self.simulator_pool.put_cts([[c, t_ind] for _, c, t_ind in epoch_cts])
                cur_num = 0
                # begin epoch
                for i_batch, batch in enumerate(self.simulator_pool):
                    # batched camera-captured 2-d images rendered by the simulators
                    num_data = len(batch)
                    end_num = cur_num + num_data
                    assert num_data, print(num_data)
                    if end_num > len(epoch_cts):
                        print('end_num {}, epoch_cts_len {}'.format(end_num, len(epoch_cts)))
                        break
                    c_inds, t_inds = zip(*[(item[0], item[2]) for item in epoch_cts[cur_num:end_num]])
                    detboxes = [self.target_model.get_detbox(image) for image in batch]
                    score_label_batch = np.array([detbox.score for detbox in detboxes])[:, None]
                    camous_batch = np.stack([epoch_cts[_num][1] for _num in range(cur_num, end_num)])

                    for c_ind, t_ind, score, image in zip(c_inds, t_inds, score_label_batch, batch):
                        self.camous_score_history[c_ind].update(score[0])
                        self.transform_scores[t_ind].update(score[0])
                        detected[c_ind] += score[0]>0.5
                    if (i_batch + 1) % self.print_every == 0:
                        self.logger.info(("Alter {}/Batch {} Avg score {:.4f}").format(i_iter, i_batch, np.mean(score_label_batch)))
                    cur_num = end_num

                # ---- ALTERNATION 2: attack using current clone network ----
                score_list = [score.avg for score in self.camous_score_history]
                print(np.shape(detected))
                precise = np.asarray(detected) / self._num_transforms
                remark_list = ['rand_srch_{}'.format(i_iter)] * len(self.camous_score_history)
                self.logger.info(
                    ("Alter:{} Batches:{} Avg score:{:.4f} P0.5:{:.4f}").format(i_iter, len(self.camous_score_history), np.mean(score_list), np.mean(precise)))
                if self.test_random:
                    exit()
                new_camous_batch, new_scores_batch = self.update_solid_camous(self.camous_history, score_list, remark_list, extract_num=self.new_camous_to_attack,)
                self._save_data_and_network(os.path.join(self.save_dir, 'Iter_{}_before_atk'.format(i_iter)))
            # new_camous_batch, new_scores_batch = self.extract_from_history(save_num=self.new_camous_to_attack,
            #                                                                valid=False, outer_iter=i_iter)
            # self.simulator.render_camous_trace(new_camous, index=0, clear=True)
            width_search = 20
            depth_search = 3
            step = 5
            for i_camous, new_camous in enumerate(new_camous_batch):
                init_avg_score = new_scores_batch[i_camous]
                init_camous = new_camous.copy()
                attack_trace = []
                self.logger.info("Attacking [{:d}/{:d}] camou, init score: {}".format(i_camous+1, len(new_camous_batch), init_avg_score))
                direct = 1
                width_init_camous = init_camous.copy()
                width_init_score = init_avg_score
                width_avg_scores, width_camous, tmp_best_score, tmp_best_camous = None, None, None, None
                for i_depth in range(depth_search):
                    if i_depth > 0:
                        if width_init_score <= tmp_best_score:
                            direct = 1
                            print('Best camous not change!')
                        else:
                            # TODO: this is important!! How to update the direction, instead of using the best, we can use all of them
                            # use max norm
                            # direct = tmp_best_camous - width_init_camous
                            # direct = direct / np.max(direct)

                            # use sign
                            direct = np.sign(tmp_best_camous - width_init_camous)

                        width_init_camous = tmp_best_camous.copy()
                        width_init_score = tmp_best_score
                    start_time = time.time()
                    self.logger.info('Begin width search for depth:{} ...'.format(i_depth))
                    for i_width in range(width_search):
                        direct *= np.random.rand(self.camous_size, self.camous_size, 3)
                        new_camous = width_init_camous + direct * step
                        new_camous.clip(min=self.bounds[0], max=self.bounds[1], out=new_camous)

                        images = self.simulator_pool.get_images([(new_camous, t_ind) for t_ind in range(self._num_transforms)])
                        detboxes = [self.target_model.get_detbox(image) for image in images]
                        scores = np.array([detbox.score for detbox in detboxes])[:, None]
                        misdetect = (scores < 0.5).sum()
                        avg_score = scores.mean()

                        diff = new_camous - init_camous
                        linf_dist = np.max(np.abs(diff))
                        l2_dist = np.linalg.norm(diff)
                        attack_trace.append([new_camous.copy(), avg_score.copy(), misdetect, l2_dist, linf_dist,
                                             'c{}_d{}_w{}'.format(i_camous, i_depth,i_width)])
                        print('{:.4f}'.format(avg_score), end=' ')
                    print()

                    width_avg_scores = np.asarray([ att[1] for att in attack_trace[-width_search:]])
                    width_camous = np.asarray([ att[0] for att in attack_trace[-width_search:]]) # width * h * w * 3
                    linf_avg_score = np.mean([att[4] for att in attack_trace[-width_search:]])

                    attack_trace = sorted(attack_trace, key=lambda x: x[1])
                    tmp_best_camous, tmp_best_score, m, l2, linf, r = attack_trace[0]

                    self.logger.info(("Camou:{} Depth:{} Best:{:.4f} misdetect:{}/{}; linf:{:.2f} "
                                "Recent:{:.4f} linf:{:.2f}").format(
                        i_camous, i_depth, tmp_best_score, m, self._num_transforms, linf, np.mean(width_avg_scores), linf_avg_score))
                    # self.logger.info('{} width search {} transforms time used {:.1f}s'.format(width_search, self._num_transforms, time.time()-start_time))

                attack_trace = sorted(attack_trace, key=lambda x: x[1])
                save_c = [att[0] for att in attack_trace]
                save_s = [att[1] for att in attack_trace]
                save_r = [att[5] for att in attack_trace]
                self.update_solid_camous(save_c, save_s, save_r, solid_max=self.solid_max)

            new_camou_list = [his[0] for his in self.solid_history]
            self.update_camous_history(new_camous=new_camou_list, saliency_mask=None, clear_all=True)
            self._save_data_and_network(os.path.join(self.save_dir, 'Iter_{}_after_atk'.format(i_iter)))

    def _save_data_and_network(self, output_dir=None):
        output_dir = os.path.join(self.save_dir, 'checkpoints') if output_dir is None else output_dir
        _mkdir_or_exists(output_dir)
        data = dict(
            solid_history=self.solid_history
        )
        mmcv.dump(data, os.path.join(output_dir, 'camous_data.pkl'))
        self.logger.info('Saving all the data to direction [{}]'.format(output_dir))
        return

    def _load_data_and_network(self, load_dir):
        assert load_dir is not None, print('Load_dir cannot be None! ')
        data = mmcv.load(load_dir)
        self.solid_history = data['solid_history'][:self.solid_max]
        # new_camous_batch, new_scores_batch = self.update_solid_camous([], [], [])
        self.one_stream(camous=self.solid_history[0][0], save_dir='solid_{:.4f}'.format(self.solid_history[0][1]))
        self.logger.info('Loading all the data from direction [{}]'.format(load_dir))
        skip = True if 'before' in load_dir else False
        load_iter = int(load_dir.split('_')[1])
        load_iter = load_iter if 'before' in load_dir else load_iter + 1
        if skip:
            self.logger.info("Skipping the fisrt loop before attack!")
        else:
            self.logger.info("Don't Skip the fisrt loop before attack, so updating camous history!")
            new_camou_list = [his[0] for his in self.solid_history]
            self.update_camous_history(new_camous=new_camou_list, saliency_mask=None, clear_all=True)
        return skip, load_iter

    def update_camous_history(self, new_camous, saliency_mask=None, clear_all=False):
        if not isinstance(new_camous, list):
            new_camous = [new_camous]
        if clear_all:
            self.camous_history = []
            self.valid_camous_history = []

        self.camous_weights = [1.] * len(self.camous_history)
        self.camous_weights = self.camous_weights + [1.] * self.num_add_camous

        for camou in new_camous:
            if saliency_mask is None:
                saliency_mask = np.ones(camou.shape)

            rand_steps = np.random.rand(self.num_add_camous - 1, self.camous_size, self.camous_size, 3) # rand means all pos
            rand_new_camous = camou + self.camous_std * saliency_mask * rand_steps
            self.camous_history += [camou]
            self.camous_history += list(np.clip(rand_new_camous, a_min=self.bounds[0], a_max=self.bounds[1]))
            self._expand_index_history(self.num_add_camous)

            if len(self.camous_history) > self.num_max_camous:
                self.camous_history = self.camous_history[-self.num_max_camous:]
                self.camous_weights = self.camous_weights[-self.num_max_camous:]
        self.logger.info("Updating camous history, {} in total".format(len(self.camous_history)))


    def eval_solid_camous(self, eval_iters=3, eval_num=5):
        self.solid_history = sorted(self.solid_history, key=lambda x: x[1])
        eval_scores = np.zeros(len(self.solid_history))
        eval_precise = np.zeros(len(self.solid_history))
        eval_num = min(eval_num, len(self.solid_history))
        for eval_i in range(eval_iters):
            for i in range(eval_num):
                save_dir = './eval_solid/No{}_time{}'.format(i, eval_i + 1)
                avg_score, precise = self.one_stream(camous=self.solid_history[i][0], save_dir=save_dir)
                eval_scores[i] = (eval_i*eval_scores[i] + avg_score)/(eval_i+1)
                eval_precise[i] = (eval_i*eval_precise[i] + precise)/(eval_i+1)
                print('No{} saved: {:.4f}, evaluated: {:.4f}, {:.4f}'.format(i+1, self.solid_history[i][1], eval_scores[i], eval_precise[i]))


    def update_solid_camous(self, camous_list, scores_list, remark_list, extract_num=0, solid_max=20):
        for camou, score, remark in zip(camous_list, scores_list, remark_list):
            if score == 0:
                print('skip {} zero score!'.format(remark))
                continue
            self.solid_history.append([camou.copy(), score, remark])
        # sorted, low score in the front
        save_num = min(len(self.solid_history), solid_max)
        self.solid_history = sorted(self.solid_history, key=lambda x: x[1])[:save_num]
        for i in range(save_num):
            avg_score, precise = self.one_stream(camous=self.solid_history[i][0], save_dir='solid_{:.4f}'.format(self.solid_history[0][1]))
            print('saved: {:.4f}, evaluated: {:.4f}'.format(self.solid_history[i][1], avg_score))
            self.solid_history[i][1] = self.solid_history[i][1]*0.7 + avg_score *0.3
        self.solid_history = sorted(self.solid_history, key=lambda x: x[1])

        extract_num = min(len(self.solid_history), extract_num)
        if extract_num > 0:
            camous_batch = []
            scores_batch = []
            log_text = 'Extracting {:d} camous with lowest detect scores: '.format(extract_num)
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


def main(device, model_file, detector, sim_cfgs, attacker_cfg, save_dir, display_info, load_clone_net, gen_data_only, train=False):
    assert sim_cfgs
    # config tf
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)

    detect_model = detector(session)
    logging.info("Constructed detect model")
    attacker = Attacker(sim_cfgs, detect_model, session, save_dir, display_info, train, **attacker_cfg)
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

faster_model_path = '/home/foxfi/unreal/tf-faster-rcnn/data/voc_2007_trainval+voc_2012_trainval/res101_faster_rcnn_iter_110000.ckpt'
light_head_model_path = "model_dump/epoch_20.ckpt"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_file")
    parser.add_argument("--display", default=":9")
    parser.add_argument("--device", default="0")
    parser.add_argument("--detect-model", default="model_dump/epoch_20.ckpt")
    parser.add_argument("--detect-model-name", default="faster_rcnn")
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

    if 'light_head_rcnn' in args.detect_model_name:
        from light_head_rcnn_model import LightHeadRCNN
        detector = LightHeadRCNN
        args.detect_model = light_head_model_path
    elif 'faster_rcnn' in args.detect_model_name:
        from faster_rcnn_model import FasterRCNN
        detector = FasterRCNN
        args.detect_model = faster_model_path
    else:
        raise NotImplementedError
    with open(args.cfg_file) as cf:
        cfgs = yaml.safe_load(cf)
    main(args.device, args.detect_model, detector, cfgs["simulator_cfg"], cfgs["attacker_cfg"],
         args.save_dir, args.display_info, args.load_clone_net, args.gen_data_only)
