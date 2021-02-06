import os
import os.path as osp
import sys
import glob

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def remove_path(keywords):
    for path in sys.path:
        if keywords in path:
            sys.path.remove(path)

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

carla_path = '/home/foxfi/unreal/carla_dynamic/PythonAPI/carla/dist'
add_path(carla_path)

faster_rcnn_path = '/home/foxfi/unreal/tf-faster-rcnn/lib'
add_path(faster_rcnn_path)

keywords = 'light_head_rcnn'
remove_path(keywords)
