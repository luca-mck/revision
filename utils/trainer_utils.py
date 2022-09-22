"""Objects added to the Trainer Class."""

from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

def get_logger(logger_config):
    return TensorBoardLogger(**logger_config)

def get_profiler(profiler_config):
    return SimpleProfiler(**profiler_config)

def get_callbacks(callback_config):
    callbacks = []
    for callback_name in callback_config:
        if callback_name == "model_checkpoint":
            callbacks.append(ModelCheckpoint(**callback_config["model_checkpoint"],))
    return callbacks