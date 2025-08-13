import os
import yaml
import gc
import torch


def load_config(config_name, config_dir="config"):
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
    config_path = os.path.join(config_dir, config_name)
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        return config["config"]
    return config


def get_arg_or_config(arg_val, config_val, default_val):
    if arg_val is not None:
        return arg_val
    if config_val is not None:
        return config_val
    return default_val


def clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
