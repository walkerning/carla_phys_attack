import carla
import Queue as queue
from carla_sim import CarlaDocker

class CarlaSyncMode(object):
    def __init__(self, world, *sensors):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self._queues = []
        self.docker = CarlaDocker('lala', 2000, 0, image_name='miao')
        self.container_id = self.docker.run()
        print('here')

    def __enter__(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.frame = self.world.apply_settings(settings)

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, type, value, traceback):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

    def _retrieve_data(self, queue, timeout):
        while True:
            data = queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

if __name__ == "__main__":
    import sys
    import time
    docker = CarlaDocker('lala', 2000, 0, image_name="carla_dynamic_vgl2")
    container_id = docker.run()

    # client = carla.Client('localhost', 2000)
    # client.set_timeout(2.0)
    num_retry = 0
    while 1:
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(1.0)
        except RuntimeError:
            if num_retry >= 10:
                print("ERROR: cannot connect to the carla RPC server after {} retries. ABORT.".format(num_retry))
                sys.exit(1)
            print("waiting and retrying ...")
            num_retry += 1
        else:
            client.set_timeout(2.0)
            break
    world = client.get_world()

    sensors = []

    try:
        sensors.append(world.spawn_actor(
            world.get_blueprint_library().find('sensor.camera.rgb'),
            carla.Transform()))
        sensors.append(world.spawn_actor(
            world.get_blueprint_library().find('sensor.camera.depth'),
            carla.Transform()))
        sensors.append(world.spawn_actor(
            world.get_blueprint_library().find('sensor.camera.semantic_segmentation'),
            carla.Transform()))

        with CarlaSyncMode(world, *sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
                snapshot = data[0]
                for n, item in enumerate(data[1:]):
                    item.save_to_disk('_out/%01d_%08d' % (n, sync_mode.frame))

    finally:
        for sensor in sensors:
            sensor.destroy()