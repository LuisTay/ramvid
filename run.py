import torch.distributed as dist
import numpy as np
import torch as th
from utils import dist_util, logger
from utils.video_datasets import load_data
from utils.resample import create_named_schedule_sampler
from utils.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from model import RAMVID
import yaml

dist_util.setup_dist()
if __name__ == "__main__":
    # Read the YAML file
    with open('model_config.yml', 'r') as stream:
        loader_value = yaml.safe_load(stream)
    # Data Loading parameters
    data_dir = loader_value.get('data_loader').get('data_dir')
    batch_size = loader_value.get('data_loader').get('batch_size')
    image_size = loader_value.get('data_loader').get('image_size')
    class_cond = loader_value.get('data_loader').get('class_cond')
    deterministic = loader_value.get('data_loader').get('deterministic')
    rgb = loader_value.get('data_loader').get('rgb')
    seq_len = loader_value.get('data_loader').get('seq_len')

    logger.log("creating data loader...")

    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=deterministic,
        rgb=rgb,
        seq_len=seq_len
    )

    RAMVID(data, loader_value).run()

