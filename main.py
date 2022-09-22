"""Entry point for object detection pipeline."""
import os 
import json

from pytorch_lightning import Trainer

from modules.data_module import ObjectDetectionDataModule
from modules.lightning_module import ObjectDetectionModel
from utils.trainer_utils import get_callbacks, get_logger, get_profiler

CONFIG_PATH = "./config/config_file.json"

def main(config_path):
    with open(config_path, "r") as fh:
        config = json.load(fh)

    # add transformations
    # add iou score metric
    # add

    datamodule = ObjectDetectionDataModule(**config["data"])
    #datamodule.prepare_data()
    #datamodule.setup()
    trainer = Trainer(
        logger=get_logger(config["logger"]),
        callbacks=get_callbacks(config["callbacks"]),
        profiler=get_profiler(config["profiler"]),
        **config["trainer"]
    )
    
    model = ObjectDetectionModel(**config["hparams"])

    trainer.fit(
        model=model,
        datamodule=datamodule
    )

    trainer.predict(ckpt_path="best", dataloaders=datamodule.val_dataloader())
    #trainer.predict(model, dataloaders=datamodule.val_dataloader())
    
    # test on validation set

if __name__ == "__main__":
    main(CONFIG_PATH)