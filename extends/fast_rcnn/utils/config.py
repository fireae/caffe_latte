#encoding:utf-8
import json
from easydict import EasyDict as edict
import os

def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = edict(config_dict)
    return config, config_dict

def process_config(json_file):
    config, _  = get_config_from_json(json_file)
    # config.summary_dir = os.path.join('../experiments', config.exp_name, 'summary/')
    # config.checkpoint_dir = os.path.join('../experiments', config.exp_name, 'checkpoint/')
    return config

cfg = process_config('utils/cfg.json')

if __name__ == '__main__':
    # cfg = process_config('cfg.json')
    print(cfg)