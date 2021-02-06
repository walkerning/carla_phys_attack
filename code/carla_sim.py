#!/usr/bin/env python
from __future__ import print_function
from colorprint import *

import os
import sys
import time
import atexit
import random
import shutil
import weakref
import subprocess

import pygame
import carla
from PIL import Image
from carla import ColorConverter as cc
import numpy as np
try:
    import queue
except ImportError:
    import Queue as queue

class CarlaDocker(object):
    def __init__(self, simulator_id, port, gpu, image_name, use_docker=True):
        self.gpu = gpu
        self.port = port
        self.use_docker = use_docker
        self.image_name = image_name
        self.simulator_id = simulator_id
        self.container_name = "carla_docker_{}".format(self.simulator_id)
        self.container_id = None
        home_dir = os.path.expanduser("~") #/home/foxfi
        self.work_dir = home_dir + '/unreal/carla_dynamic/'
        _texture_dir = os.path.join(home_dir, "carla_data/data_{}".format(self.simulator_id))
        self._volume_mappings = {
            "/tmp/.X11-unix": "/tmp/.X11-unix",
            "/home/carla/.Xauthority": os.path.join(home_dir, ".Xauthority"),
            "/home/carla/data": _texture_dir
        }


    def run(self):
        if self.use_docker:
            cmd = "docker run --rm -d -p {port}-{end_port}:{port}-{end_port} --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES={gpu} --name {name} ".format(port=self.port, end_port=self.port+2, gpu=self.gpu, name=self.container_name) + \
                " ".join(["-v {}:{}".format(h_path, c_path) for c_path, h_path in self._volume_mappings.items()]) + " {} ./CarlaUE4.sh -world-port={}".format(self.image_name, self.port)
            print("RUN docker: {}".format(cmd))
            outputs = subprocess.check_output(cmd, shell=True)
            self.container_id = outputs.strip()
        else:
            cmd = "bash {}CarlaUE4.sh -opengl4 -world_port={}".format(self.work_dir, self.port)
            print("RUN Command: {}".format(cmd))
            outputs = subprocess.Popen(cmd, shell=True)
            self.container_id = 0
        # TODO: use logging!

        # self.container_id = 0
        return self.container_id

    def stop(self):
        if self.container_id is not None:
            cmd = "docker container stop {}".format(self.container_name)
            print("STOP docker: {}".format(cmd))
            subprocess.check_call(cmd, shell=True)
        self.container_id = None

# TODO: can we get segmentation info from carla?
class CarlaSimulator(object):
    def __init__(self, gpu=2, host="127.0.0.1", port=5000, already_started=False, use_docker=False,
                 image_size=(360, 240), docker_image_name="carla_dynamic_vgl2", simulator_id=None):
        # configs image_size: w*h
        self.host = host
        self.port = port
        self.gpu = gpu
        self.image_size = image_size
        self.already_started = already_started
        self.use_docker = use_docker
        self.frame = None
        self.extra_ticks = 30
        # self.frame=0

        # get a random simulator id if not specified
        if simulator_id is None:
            self.simulator_id = "{:08x}".format(random.getrandbits(32))
        else:
            self.simulator_id = simulator_id

        # create the texture dir

        home_dir = os.path.expanduser("~")
        self._texture_dir = os.path.join(home_dir, "carla_data/data_{}".format(self.simulator_id))
        self._texture_path = os.path.join(self._texture_dir, "tesla.png")
        if not os.path.exists(self._texture_dir):
            os.makedirs(self._texture_dir)

        # run the docker
        if not self.already_started:
            print("Start carla docker, simulator id: {}".format(self.simulator_id))
            self.docker = CarlaDocker(self.simulator_id, self.port, self.gpu, image_name=docker_image_name, use_docker=self.use_docker)
            self.container_id = self.docker.run()

        self._cleanuped = False
        atexit.register(self.cleanup)

        print(("Waiting for the initialization of carla simulator {}, "
               "it could take up to tens of seconds ...").format(self.simulator_id))
        init_start_time = time.time()
        num_retry = 0
        while 1:
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(10.0)
                world = self.client.get_world()
            except RuntimeError:
                if num_retry >= 10:
                    print("ERROR: cannot connect to the carla RPC server after {} retries. ABORT.".format(num_retry))
                    sys.exit(1)
                print("waiting and retrying ...")
                num_retry += 1
            else:
                self.client.set_timeout(2.0)
                break
        init_time = time.time() - init_start_time
        print("Finish initialization of carla simulator {}: elapsed {:.2f}s".format(self.simulator_id, init_time))

        # fix weather for now
        weather = carla.WeatherParameters(cloudyness=10.0, precipitation=0.0, sun_altitude_angle=70.0)
        world.set_weather(weather)
        self.ori_settings = world.get_settings()
        self.delta_seconds = 1.0 / 20
        settings = carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds)
        # stuck here
        world.apply_settings(settings)

        self.world = world
        bp_lib = world.get_blueprint_library()
        blueprints = bp_lib.filter("vehicle.*")
        self.tesla_blueprint = blueprints[18]
        print(self.tesla_blueprint)

        # spawn_points = world.get_map().get_spawn_points()
        self.location = carla.Location(x=-6.45, y=-79.06, z=1.84)
        self.rotation = carla.Rotation(yaw=92.0)

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        Attachment = carla.AttachmentType

        # spawn vehicle
        transform = carla.Transform(self.location, self.rotation)
        command_batch = [SpawnActor(self.tesla_blueprint, transform).then(SetAutopilot(FutureActor, False))]

        for response in self.client.apply_batch_sync(command_batch):
            if response.error:
                print("Error: ", response.error)
            else:
                self.ego_vehicle_id = response.actor_id
        print("ego vehicle id: ", self.ego_vehicle_id)
        self.ego_vehicle = self.world.get_actor(self.ego_vehicle_id)


        self.image_queue = queue.Queue()

        # create sensor
        self._camera_transform = carla.Transform(carla.Location(x=-4.5, z=1.5), carla.Rotation(pitch=8.0))
            # (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
        # (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
        # (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
        # (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_size[0]))
        camera_bp.set_attribute("image_size_y", str(self.image_size[1]))
        # camera_bp.set_attribute('sensor_tick', '1.0')
        self.camera = self.world.spawn_actor(
            camera_bp,
            self._camera_transform,
            attach_to=self.ego_vehicle,
            attachment_type=Attachment.Rigid
        )
        print('start listening', color='green')
        self.last_camera_frame = 0
        self.camera.listen(lambda image: CarlaSimulator._parse_image(weakref.ref(self), image))
        # import ipdb
        # ipdb.set_trace()
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.image_size[0] * 2, self.image_size[1] * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.info_dim = (self.image_size[0], self.image_size[1] / 2)
        self.info_surface = pygame.Surface(self.info_dim)
        self.info_surface.set_alpha(100)
        self.clock = pygame.time.Clock()

        # for display info
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._old_info_text = None

        # attributes
        self.surface = None
        self.last_image = None
        self.world.on_tick(self.on_world_tick) # tick and render pygame

        # tick for 20 times first
        num_init_ticks = 0
        while num_init_ticks < 20:
            self.world.tick()
            num_init_ticks += 1
        print('CarlaSimulator initialized, ')

    def on_world_tick(self, timestamp):
        self.clock.tick()
        self.sim_time = timestamp.elapsed_seconds
        self.render()
        # print("finished rendering start sim:{}. end sim {}, sys:{}".format(self.sim_time, timestamp.elapsed_seconds, time.time()))

    @staticmethod
    def _parse_image(weak_self, image):
        # image.save_to_disk('output/%06d.png' % image.frame)
        # print(image.frame)
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.image_queue.put((array, image.frame))
        self.last_camera_frame = image.frame
        print('parse:',image.frame)
        # print('_parse_image! self.image_queue.qsize: {} , put frame: {}'.format(self.image_queue.qsize(), image.frame))

    def render_camous_trace(self, camous_trace, index=None, clear=False):
        if index is not None:
            assert isinstance(camous_trace, np.ndarray)
            if clear:
                # clear old camous trace
                clear_surface = pygame.Surface((self.image_size[0], self.image_size[1] * 2))
                clear_surface.fill((0,0,0))
                self.display.blit(clear_surface, [0, self.image_size[1]])
            camous = camous_trace
            # assume all camous have same shape 
            h_offset = 4 + (camous_trace.shape[1] + 4) * index
            camous_surface = pygame.surfarray.make_surface(camous.swapaxes(0, 1))
            self.display.blit(camous_surface, (h_offset, self.image_size[1] + 8))
            return

        h_offset = 4 
        for camous in camous_trace:
            camous_surface = pygame.surfarray.make_surface(camous.swapaxes(0, 1))
            self.display.blit(camous_surface, (h_offset, self.image_size[1] + 8))
            h_offset += camous.shape[1] + 4 # height x width x channel

    def render(self, camous=None, info=None, box=[]):
        if self.surface is not None:
            self.display.blit(self.surface, (0, 0))
            if camous is not None:
                camous = np.zeros(camous.shape)
                camous_surface = pygame.surfarray.make_surface(camous.swapaxes(0, 1))
                self.display.blit(camous_surface, (self.image_size[0], 0))
            # display info surface
            self.display.blit(self.info_surface, (0, 0))
            if info is not None:
                self.render_info(box, info)
            elif self._old_info_text is not None:
                # keep the old info here
                self.render_info(box, self._old_info_text)
            else:
                self.render_info(box, ['here to print text'])
        pygame.display.update()

    def render_info(self, boxes, info_text):
        self._old_info_text = info_text
        v_offset = 4
        for box in boxes:
            rect_border = pygame.Rect(box)
            pygame.draw.rect(self.display, (255, 0, 0), rect_border, 1)
        for item in info_text:
            if v_offset + 18 > self.info_dim[1]:
                break
            if item: # str
                surface = self._font_mono.render(item, True, (255, 255, 255))
                self.display.blit(surface, (8, v_offset))
            v_offset += 18
        pygame.display.update()

    def cleanup(self):
        if self._cleanuped:
            # avoid called multiple times
            return
        print("clean up the carla simulator", color='yellow')
        if hasattr(self, "world"):
            self.world.remove_on_tick(id(self.on_world_tick))
        if os.path.exists(self._texture_dir):
            shutil.rmtree(self._texture_dir)
        if hasattr(self, "client") and hasattr(self, "ego_vehicle_id"):
            self.client.apply_batch([carla.command.DestroyActor(self.ego_vehicle_id)])
        if hasattr(self, "world") and hasattr(self, "ori_settings"):
            self.world.apply_settings(self.ori_settings)
        if hasattr(self, "docker"):
            self.docker.stop()
        self._cleanuped = True

    def get_image(self, camous, transform, timeout=2, tick=3):
        if camous is None:
            # use original texture
            if os.path.exists(self._texture_path):
                os.unlink(self._texture_path)
        else:
            camous = _magnify_camous(camous, repeat_times=4, enlarge_times=int(512/camous.shape[0]))
            im = Image.fromarray(camous.astype(np.uint8))
            im.save(self._texture_path)
            im.save('./data/lateset_camous.jpg')
        if transform is not None:
            self.camera.set_transform(transform)
        # FIXME: async call to `set_transform`, needed synchronized call
        for _ in range(tick):
            self.frame = self.world.tick()
        for i in range(self.extra_ticks):
            if self.last_camera_frame >= self.frame -i:
                # print('last frame:', self.last_camera_frame)
                break
            self.frame = self.world.tick()
            # if self.frame == self.last_camera_frame:
            #     break
            # raise RuntimeError
        # print('get image! simulater ID:{} self.image_queue: {}'.format(self.simulator_id, self.image_queue.qsize()))
        self.last_image = self._retrieve_data(self.image_queue, timeout)
        if self.last_image is None:
            return None
        # to render on pygame display
        self.surface = pygame.surfarray.make_surface(self.last_image.swapaxes(0, 1))
        loc = transform.location
        rot = transform.rotation
        trans_info = 'Location Into x:{:.1f}, y:{:.1f}, z:{:.1f}, rot:{:.1f}'.format(loc.x, loc.y, loc.z, rot.yaw)
        self.render(camous, [trans_info])
        return self.last_image

    def _retrieve_data(self, sensor_queue, timeout):
        try_times = 0
        while True:
            try:
                data, frame = sensor_queue.get(timeout=timeout)
                # print('get frame {}, self.frame {} self.image_queue.qsize {}'.format(frame, self.frame, sensor_queue.qsize()))
            except queue.Empty:
                try_times += 1
                print('---- failed once ----')
                if try_times > 3:
                    # self.cleanup()
                    print('simulator {} stopped because of empty queue'.format(self.simulator_id))
                    # raise
                    return None
                continue
            if frame >= self.frame - self.extra_ticks:
                # print('get frame == self.frame, return')
                return data.copy()

def _magnify_camous(camous, save_name=None, save_size=2048, repeat_times=4, enlarge_times=32):
    camous_size = camous.shape[0]
    assert save_size % camous_size == 0
    assert repeat_times*enlarge_times*camous_size == save_size
    large_camous = np.zeros((enlarge_times*camous_size, enlarge_times*camous_size, 3))
    # TODO: more efficient
    p = enlarge_times
    for i in range(camous_size):
        for j in range(camous_size):
            for k in range(3):
                large_camous[p * i:p * i + p, p * j:p * j + p, k] = camous[i, j, k]
    p = repeat_times
    repeat_1 = large_camous
    for _ in range(p - 1):
        repeat_1 = np.concatenate((repeat_1, large_camous), axis=0)
    repeat_2 = repeat_1
    for _ in range(p - 1):
        repeat_2 = np.concatenate((repeat_2, repeat_1), axis=1)
    visual_camous = repeat_2
    if save_name is not None:
        _save_image(visual_camous, save_name)
    return visual_camous.copy()

def _save_image(im, path):
    root_path = os.path.dirname(path)
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    Image.fromarray(im.astype(np.uint8)).save(path)

if __name__ == '__main__':
    os.environ["DISPLAY"] = ":9"
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
        carla.Transform(carla.Location(y=5.0, x=-5.0, z=1.0), carla.Rotation(yaw=-45)),
        # carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
        # carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)),
        # carla.Transform(carla.Location(x=-1, y=-3, z=0.5))
    ]
    try:
        sim = CarlaSimulator(already_started=False)
        # tick for 20 times first
        num_init_ticks = 0
        while num_init_ticks < 20:
            sim.world.tick()
            num_init_ticks += 1

        print("Start rendering", color='green')
        # for i in range(3):
        new_camous = 255 * np.random.rand(16, 16, 3)
        big_camous = _magnify_camous(new_camous,  "./tests_new/big_camous.png", 2048)
        print('big!')
        big_camous = new_camous
        for i, transform in enumerate(transforms):
            # rendered_im = sim.get_image(None, transform)
            rendered_im = sim.get_image(new_camous, transform)
            rendered_im_2 = sim.get_image(big_camous, transform)
            # _save_image(new_camous, "./tests_new/test_camous_{}.png".format(i))
            _save_image(rendered_im, "./tests_new/test_trans_rendered_{}.png".format(i))
            _save_image(rendered_im_2, "./tests_new/test_trans_rendered_big_{}.png".format(i))
        print('done 255', color='green')
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        pass
    finally:
        if "sim" in globals():
            sim.cleanup()
        pygame.quit()
        print('\ndone.')
        sys.exit(0)
