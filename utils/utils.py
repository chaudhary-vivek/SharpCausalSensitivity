import os
import yaml
import torch
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pytorch_lightning.loggers.neptune import NeptuneLogger
import torch.nn.functional as fctnl
from torch.utils.data import DataLoader
import random
import numpy as np


def get_device():
    if torch.cuda.is_available():
        gpu = 1
    else:
        gpu = 0
    return gpu


def get_device_string():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_project_path():
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    return str(path.parent.absolute())


def load_yaml(path_relative):
    return yaml.safe_load(open(get_project_path() + path_relative + ".yaml", 'r'))


def save_yaml(path_relative, file):
    with open(get_project_path() + path_relative + ".yaml", 'w') as outfile:
        yaml.dump(file, outfile, default_flow_style=False)


def save_pytorch_model(path_relative, model):
    torch.save(model.state_dict(), get_project_path() + path_relative + ".pt")


def load_pytorch_model(path_relative, model):
    model.load_state_dict(torch.load(get_project_path() + path_relative + ".pt"))
    return model


def get_logger(neptune=True):
    if neptune:
        logger = NeptuneLogger(project='dennisfrauen/sharp-sensitivity')
    else:
        logger = True
    return logger


def get_config_names(model_configs):
    config_names = []
    for model_config in model_configs:
        config_names.append(model_config["name"])
    return config_names


def mse_bce(y, y_hat, y_type="continuous"):
    if y_type == "continuous":
        return torch.mean((y - y_hat) ** 2)
    if y_type == "binary":
        return fctnl.binary_cross_entropy(y_hat, y, reduction='mean')


def train_model(model, config, d_train, d_val=None, savepath_rel=None):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    logger = get_logger(config["neptune"])

    trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False,
                         accelerator="auto", logger=logger, enable_checkpointing=False)

    train_loader = DataLoader(dataset=d_train, batch_size=batch_size, shuffle=True)
    if d_val is not None:
        val_loader = DataLoader(dataset=d_val, batch_size=batch_size, shuffle=False)
        try:
            trainer.fit(model, train_loader, val_loader)
            val_results = trainer.validate(model=model, dataloaders=val_loader, verbose=False)
        except:
            print("Model training + validation failed, returning large validation error")
            val_results = [{"val_obj": 1000000}]
    else:
        trainer.fit(model, train_loader)
        val_results = None
    #Save model
    if savepath_rel is not None:
        save_pytorch_model(savepath_rel, model)
    return {"trained_model": model, "val_results": val_results}
