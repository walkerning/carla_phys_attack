#!/usr/bin/env python
from __future__ import print_function
import os
import atexit
import weakref
import pygame
import carla
from PIL import Image
from carla import ColorConverter as cc
import numpy as np
try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSimulator(object):
    def __init__(self, host="127.0.0.1", port=2000, image_size=(256, 256)):
        # configs
        self.host = host
        self.port = port
        self.image_size = image_size

        # spawn the vehicle
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(2.0)
        world = self.client.get_world()
        # fix weather for now
        weather = carla.WeatherParameters(cloudyness=10.0, precipitation=0.0, sun_altitude_angle=70.0)
        world.set_weather(weather)
        self.ori_settings = world.get_settings()
        self.delta_seconds = 1.0 / 20
        settings = carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds)
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

        # create sensor
        self._camera_transform = carla.Transform(carla.Location(x=-4.5, z=1.5), carla.Rotation(pitch=8.0))
            # (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
        # (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
        # (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
        # (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.image_size[0]))
        camera_bp.set_attribute("image_size_y", str(self.image_size[1]))
        self.camera = self.world.spawn_actor(
            camera_bp,
            self._camera_transform,
            attach_to=self.ego_vehicle,
            attachment_type=Attachment.Rigid
        )
        self.camera.listen(lambda image: CarlaSimulator._parse_image(weakref.ref(self), image))

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

        atexit.register(self.cleanup) # clean up

        # attributes
        self.surface = None
        self.last_image = None
        self.world.on_tick(self.on_world_tick) # tick and render pygame
        self.image_queue = queue.Queue()

        # tick for 20 times first
        num_init_ticks = 0
        while num_init_ticks < 20:
            self.world.tick()
            num_init_ticks += 1

    def on_world_tick(self, timestamp):
        self.clock.tick()
        self.sim_time = timestamp.elapsed_seconds
        self.render()
        # print("sim time: ", self.sim_time)

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.image_queue.put((array, image.frame))

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

    def render(self, camous=None):
        if self.surface is not None:
            self.display.blit(self.surface, (0, 0))
            if camous is not None:
                camous_surface = pygame.surfarray.make_surface(camous.swapaxes(0, 1))
                self.display.blit(camous_surface, (self.image_size[0], 0))
            # display info surface
            self.display.blit(self.info_surface, (0, 0))
            if self._old_info_text is not None:
                # keep the old info here
                self.render_info([], self._old_info_text)
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
        self.client.apply_batch([carla.command.DestroyActor(self.ego_vehicle_id)])
        self.world.apply_settings(self.ori_settings)

    def get_image(self, camous, transform, timeout=2, ori=False):
        if ori:
            # use original texture
            assert camous is None
            if os.path.exists("/home/foxfi/data/tesla.png"):
                os.unlink("/home/foxfi/data/tesla.png")
        if camous is not None:
            im = Image.fromarray(camous.astype(np.uint8))
            im.save("/home/foxfi/data/tesla.png")
        if transform is not None:
            self.camera.set_transform(transform)
        for _ in range(2):
            self.frame = self.world.tick()
        self.last_image = self._retrieve_data(self.image_queue, timeout)
        # to render on pygame display
        self.surface = pygame.surfarray.make_surface(self.last_image.swapaxes(0, 1))
        # self.world.wait_for_tick()
        self.render(camous)
        # while self.last_image is None:
        #     continue
        return self.last_image

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data, frame = sensor_queue.get(timeout=timeout)
            if frame == self.frame:
                return data

def _save_image(im, path):
    Image.fromarray(im.astype(np.uint8)).save(path)

if __name__ == '__main__':
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
        sim = CarlaSimulator()
        # tick for 20 times first
        num_init_ticks = 0
        while num_init_ticks < 20:
            sim.world.tick()
            num_init_ticks += 1

        # for i in range(3):
        for i, transform in enumerate(transforms):
            # new_camous = 255 * np.random.rand(16, 16, 3)
            rendered_im = sim.get_image(None, transform)
            # _save_image(new_camous, "./tests_new/test_camous_{}.png".format(i))
            _save_image(rendered_im, "./tests_new/test_trans_rendered_{}.png".format(i))
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        print('\ndone.')
