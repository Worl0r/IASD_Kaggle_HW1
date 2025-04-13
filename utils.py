import logging


import yaml


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dict_from_pyfile(module):
    if module is None:
        return {}

    listExcep = ["torch", "cudnn", "nn"]

    # Obtain all the variables from the module
    return {
        k: v
        for k, v in vars(module).items()
        if not callable(v) and not k.startswith("__")
        if k not in listExcep
    }

def get_logger() -> logging.Logger:
    # Create the config for the logger
    logging.basicConfig(
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    # Set the logger level of priority
    logging.getLogger().setLevel(logging.DEBUG)

    # Pick the name of the current file
    logger = logging.getLogger(__name__)

    return logger

def load_config(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)