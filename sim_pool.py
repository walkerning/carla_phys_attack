from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import atexit
import multiprocessing
import Queue as queue # Python 2

import numpy as np
from carla_sim import CarlaSimulator

def _worker_loop(sim_cfg, carla_ts, batch_ind_queue, batch_data_queue, int_ind_queue, int_data_queue, b_timeout):
    print("Start worker process: PID ", os.getpid())
    simulator = CarlaSimulator(**sim_cfg)
    print("Data worker started for simulator {}".format(simulator.simulator_id))
    while 1:
        try:
            ct = int_ind_queue.get_nowait()
            if ct is None:
                # end worker loop
                break
            data_queue = int_data_queue
        except queue.Empty:
            try:
                ct = batch_ind_queue.get(timeout=b_timeout)
            except queue.Empty:
                continue
            data_queue = batch_data_queue
        ct_idx, camous, trans = ct
        image = simulator.get_image(camous, carla_ts[trans])
        data_queue.put((ct_idx, image))
    simulator.cleanup()
    print("Data worker stop for simulator {}".format(simulator.simulator_id))
    sys.exit(0)


class SimulatorPool(object):
    def __init__(self, carla_transforms, sim_cfgs, batch_size, batch_poll_timeout=2.0):
        self.carla_transforms = carla_transforms
        self.sim_cfgs = sim_cfgs
        self.batch_size = batch_size

        self.batch_ind_queue = multiprocessing.Queue()
        self.batch_data_queue = multiprocessing.Queue()
        self.int_ind_queue = multiprocessing.Queue()
        self.int_data_queue = multiprocessing.Queue()
        self.shutdown = False
        self.int_data_idx = 0
        self.send_idx = 0
        self.recv_idx = 0
        self.reorder_dict = {}
        self.int_reorder_dict = {}
        self.workers = [
            multiprocessing.Process(
                target=_worker_loop,
                args=(sim_cfg, self.carla_transforms, self.batch_ind_queue, self.batch_data_queue,
                      self.int_ind_queue, self.int_data_queue, batch_poll_timeout))
            for sim_cfg in self.sim_cfgs]
        for w in self.workers:
            w.daemon = True
            w.start()
        atexit.register(self._shutdown)

    def put_cts(self, ct_lst):
        for c, t in ct_lst:
            self.batch_ind_queue.put((self.send_idx, c, t))
            self.send_idx += 1

    def get_images(self, ct_lst):
        # synchronize call to get images using interrupt_data/ind_queue
        # this method is not thread safe, only call it from one process
        should_return_idxes = []
        for c, t in ct_lst:
            self.int_ind_queue.put((self.int_data_idx, c, t))
            should_return_idxes.append(self.int_data_idx)
            self.int_data_idx += 1
        batch = self._get_data_from_queue(should_return_idxes, self.int_data_queue, self.int_reorder_dict)
        return np.stack(batch)

    def __iter__(self):
        return self

    def __len__(self):
        num_data = self.send_idx - self.recv_idx
        return (num_data + self.batch_size - 1) // self.batch_size

    def _get_data_from_queue(self, idxes, data_queue, reorder_dct):
        batch = []
        for idx in idxes:
            if self.shutdown:
                break
            if idx in reorder_dct:
                batch.append(reorder_dct.pop(idx))
                continue
            while not self.shutdown:
                try:
                    new_idx, image = data_queue.get(timeout=10.0)
                except queue.Empty:
                    continue
                if new_idx != idx:
                    reorder_dct[new_idx] = image
                else:
                    batch.append(image)
                    break
        return batch
                    
    def __next__(self):
        # return a batch of images
        end_idx = min(self.recv_idx + self.batch_size, self.send_idx)
        if end_idx <= self.recv_idx:
            raise StopIteration
        should_return_idxes = np.arange(self.recv_idx, end_idx)
        batch = self._get_data_from_queue(should_return_idxes, self.batch_data_queue, self.reorder_dict)
        self.recv_idx = end_idx
        return np.stack(batch)

    next = __next__

    def _shutdown(self):
        if self.shutdown:
            return
        print("Shutdown simulator workers")
        self.shutdown = True
        for i in range(len(self.sim_cfgs)):
            self.int_ind_queue.put(None)

    def __del__(self):
        self._shutdown()

if __name__ == "__main__":
    import carla
    from carla_sim import _save_image

    os.environ["DISPLAY"] = ":8"
    
    sim_cfgs = [{
        "gpu": 0,
        "port": 22000,
        #"simulator_id": "try1"
    }, {
        "gpu": 2,
        "port": 22010,
        # "simulator_id": "try2"
    }
    ]

    camous_history = [255 * np.random.rand(16, 16, 3) for _ in range(10)]
    transforms = [
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
    sim_pool = SimulatorPool(transforms, sim_cfgs, batch_size=16)

    num_camous = len(camous_history)
    num_transform = len(transforms)
    
    total_size = 32
    c_inds = np.random.randint(low=0, high=num_camous, size=total_size)
    t_inds = np.random.randint(low=0, high=num_transform, size=total_size)

    sim_pool.put_cts([(camous_history[c_ind], t_ind) for c_ind, t_ind in zip(c_inds, t_inds)])
    assert len(sim_pool) == 2
    s_time = time.time()
    batches = [batch for batch in sim_pool]
    e_time = time.time() - s_time
    print("elapsed time: {:.3f} s".format(e_time))
    assert len(batches) == 2
    assert len(batches[0]) == 16

    # try second "epoch"
    c_inds = np.concatenate([c_inds, [0] * 15])
    t_inds = np.concatenate([t_inds, np.arange(15)])
    sim_pool.put_cts([(camous_history[0], t_ind) for t_ind in range(num_transform)])
    assert len(sim_pool) == 1
    batches += [batch for batch in sim_pool]
    assert len(batches[-1]) == 15
    batches = np.concatenate(batches, axis=0)
    [_save_image(c, "test_sim_pool_images/camous_{}.jpg".format(i))
     for i, c in enumerate(camous_history)]
    [_save_image(im, "test_sim_pool_images/sim_pic_c{}_t{}.jpg".format(c_ind, t_ind))
     for im, c_ind, t_ind in zip(batches, c_inds, t_inds)]
    
    # try method `get_images`
    camous1_images = sim_pool.get_images([(camous_history[1], t_ind) for t_ind in range(num_transform)])
    assert len(camous1_images) == 15
    [_save_image(im, "test_sim_pool_images/sim_pic_c1_t{}.jpg".format(t_ind))
     for t_ind, im in enumerate(camous1_images)]

    # try render clean image (i.e. without camous)
    clean_images = sim_pool.get_images([(None, t_ind) for t_ind in range(num_transform)])
    assert len(clean_images) == 15
    [_save_image(im, "test_sim_pool_images/sim_pic_clean_t{}.jpg".format(t_ind))
     for t_ind, im in enumerate(clean_images)]
