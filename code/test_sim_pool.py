
import os
import numpy as np
import carla
from sim_pool import SimulatorPool
from carla_sim import CarlaSimulator, _save_image

os.environ["DISPLAY"] = ":8"

sim_cfgs = [{
    "gpu": 1,
    "port": 22000,
    #"simulator_id": "try1"
}# , {
#     "gpu": 2,
#     "port": 22010,
#     # "simulator_id": "try2"
# }
]

simulators = [CarlaSimulator(**sim_cfg) for sim_cfg in sim_cfgs]
sim_pool = SimulatorPool(simulators, batch_size=16)

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
num_camous = len(camous_history)
num_transform = len(transforms)

total_size = 32
c_inds = np.random.randint(low=0, high=num_camous, size=total_size)
t_inds = np.random.randint(low=0, high=num_transform, size=total_size)

sim_pool.put_cts([(camous_history[c_ind], transforms[t_ind]) for c_ind, t_ind in zip(c_inds, t_inds)])
batches = np.concatenate([batch for batch in sim_pool], axis=0)
assert len(batches) == 2
assert len(batches[0]) == 16
[_save_image(c, "test_sim_pool_images/camous_{}.jpg".format(i)) for i, c in enumerate(camous_history)]
[_save_image(im, "test_sim_pool_images/sim_pic_c{}_t{}.jpg".format(c_ind, t_ind)) for im, c_ind, t_ind in zip(batches, c_inds, t_inds)]
