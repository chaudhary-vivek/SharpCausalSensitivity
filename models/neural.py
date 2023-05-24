# Neural networks
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pyro.distributions as dist
import pyro.distributions.transforms as T


# MLP for multi-class classification (discrete conditional distributions) or continuous regression
class MLP(pl.LightningModule):
    def __init__(self, config, out_type="discrete"):
        super().__init__()
        input_size = config["d_in"]
        hidden_size = config["d_hidden"]
        self.output_size = config["d_out"]
        self.output_type= out_type
        if out_type == "discrete":
            if self.output_size == 1:
                self.loss = torch.nn.BCELoss(reduction='mean')
            else:
                self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        elif out_type == "continuous":
            self.loss = torch.nn.MSELoss(reduction='mean')
        else:
            raise ValueError("out_type must be either 'discrete' or 'continuous'")
        dropout = config["dropout"]

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.output_size),
            # nn.Softmax()
        )
        self.neptune = config["neptune"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def objective(self, batch, out):
        loss = self.loss(out, batch["y"])
        return {"obj": loss}

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Forward pass
        out = self.network(train_batch["x"])
        if self.output_size == 1 and self.output_type == "discrete":
            out = out.sigmoid()
        # Loss
        obj_dict = self.objective(train_batch, out)
        # Logging
        if self.neptune:
            obj_dict_train = dict([("train_" + key, value) for key, value in obj_dict.items()])
            self.log_dict(obj_dict_train, logger=True, on_epoch=True, on_step=False)
        return obj_dict["obj"]

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        # Forward pass
        out = self.network(val_batch["x"])
        if self.output_size == 1 and self.output_type == "discrete":
            out = out.sigmoid()
        # Loss
        obj_val = self.objective(val_batch, out)
        # Logging
        obj_dict_val = dict([("val_" + key, value) for key, value in obj_val.items()])
        self.log_dict(obj_dict_val, logger=True, on_epoch=True, on_step=False)
        return obj_val["obj"]

    def predict(self, x, scaling_params=None):
        self.eval()
        if self.output_type == "discrete":
            if self.output_size > 1:
                out = self.network(x).softmax(dim=-1)
            else:
                out = self.network(x).sigmoid()
        else:
            out = self.network(x)
            if scaling_params is not None:
                out = out * scaling_params["sd"] + scaling_params["mean"]
        return out.detach()


class CondNormalizingFlow(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        input_size = config["d_in"]
        output_size = config["d_out"]
        hidden_size = config["d_hidden"]
        count_bins = config["count_bins"]

        dist_base = dist.Normal(torch.zeros(output_size), torch.ones(output_size))
        self.y_transform = T.conditional_spline(output_size, context_dim=input_size, hidden_dims=[hidden_size, hidden_size],
                                                count_bins=count_bins)
        self.dist_y_given_x = dist.ConditionalTransformedDistribution(dist_base, [self.y_transform])

        self.neptune = config["neptune"]
        self.optimizer = torch.optim.Adam(self.y_transform.parameters(), lr=config["lr"])
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def objective(self, batch):
        x = batch["x"]
        y = batch["y"]
        # Forward pass
        ln_p_y_given_x = self.dist_y_given_x.condition(x).log_prob(y)
        return {"obj": - ln_p_y_given_x.mean()}

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Loss
        obj_dict = self.objective(train_batch)
        # Logging
        if self.neptune:
            obj_dict_train = dict([("train_" + key, value) for key, value in obj_dict.items()])
            self.log_dict(obj_dict_train, logger=True, on_epoch=True, on_step=False)
        return obj_dict["obj"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.dist_y_given_x.clear_cache()

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        # Loss
        obj_val = self.objective(val_batch)
        # Logging
        obj_dict_val = dict([("val_" + key, value) for key, value in obj_val.items()])
        self.log_dict(obj_dict_val, logger=True, on_epoch=True, on_step=False)
        return obj_val["obj"]

    #Evaluates density of y given x on a grid of y values
    #x is of shape (batch_size, d_in), y is of shape (n_grid, 1)
    def predict_density(self, x, y, scaling_params=None):
        self.eval()
        x_full = x.repeat_interleave(y.shape[0], dim=0)
        y_full = y.repeat(x.shape[0], 1)
        if scaling_params is not None:
            y_full = (y_full - scaling_params["mean"]) / scaling_params["sd"]
        pred = self.dist_y_given_x.condition(x_full).log_prob(y_full).exp()
        if scaling_params is not None:
            pred = pred / scaling_params["sd"]
        return pred

    def sample(self, x, n_samples, scaling_params=None):
        self.eval()
        # x is of shape (batch_size, d_in)
        samples = torch.squeeze(self.dist_y_given_x.condition(x).sample(torch.Size([n_samples, x.shape[0]])))
        if samples.dim() == 1:
            samples = samples.unsqueeze(1)
        samples = torch.transpose(samples, 0, 1)
        if scaling_params is not None:
            samples = samples * scaling_params["sd"] + scaling_params["mean"]
        return samples


