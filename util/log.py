from datetime import datetime
import logging
import os
import re
import torch
import yaml


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def convert_values(data):
    if isinstance(data, dict):
        return {k: convert_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_values(v) for v in data]
    else:
        return exp_to_num(data) if isinstance(data, str) else data

def exp_to_num(data):
    pattern = r"^[+-]?\d+(\.\d+)?[eE][+-]?\d+$"

    if re.match(pattern, data):
        data = float(data)
        return data if data < 1 else int(data)
    else:
        return data

def check_env():
    if torch.cuda.is_available():
        return 'cuda', [gpu for gpu in range(torch.cuda.device_count())]
    elif torch.mps.is_available():
        return 'mps', [mps for mps in range(torch.mps.device_count())]
    else:
        return 'cpu', None


def prepare(args):
    # Load config file
    config = convert_values(load_yaml(args.config))
    config['env'], config['gpu_ids'] = check_env()
    try:
        config['distributed'] = False if len(config['gpu_ids']) == 1 else True
    except TypeError:
        config['distributed'] = False
    if isinstance(args.input, list):
        config['fuse'] = True
    args.config = config
        
    # Set up experiment directory
    experiment_dir = os.path.join(args.output,f'{config['experiment_name']}_{get_timestamp()}')
    mkdirs(experiment_dir)
    args.experiment_root = experiment_dir
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(experiment_dir, 'log.txt')),
                                  logging.StreamHandler()])
    
    logging.info('Arguments:\n' + '\n'.join([f'--{k} {v}' for k, v in vars(args).items()]))

    return args