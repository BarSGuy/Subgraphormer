import torch.nn as nn
import torch
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import average_precision_score
from ogb.graphproppred import Evaluator
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import logging
import copy
import wandb

# --------------------------------- loss function -------------------------------- #


def get_loss_func(cfg):
    dataset_name = cfg.data.name

    dataset_info = {
        'zinc12k': (nn.L1Loss(), 'minimize', 'regression'),
        'zincfull': (nn.L1Loss(), 'minimize', 'regression'),

        'ogbg-molesol': (RMSELoss(), 'minimize', 'regression'),
        'ogbg-molbace': (nn.BCEWithLogitsLoss(), 'maximize', 'classification'),
        'ogbg-molhiv': (nn.BCEWithLogitsLoss(), 'maximize', 'classification'),
        'alchemy': (nn.L1Loss(), 'minimize', 'regression'),

        'Peptides-func': (nn.BCEWithLogitsLoss(), 'maximize', 'classification'),
        'Peptides-struc': (nn.L1Loss(), 'minimize', 'regression'),
    }

    if dataset_name not in dataset_info:
        raise ValueError(
            f"No loss function available for the dataset: {dataset_name}")

    return dataset_info[dataset_name]


class RMSELoss(nn.MSELoss):
    def forward(self, output, target):
        mse = super().forward(output, target)
        return torch.sqrt(mse)

# --------------------------------- optimizer -------------------------------- #


def get_optim_func(cfg, model):
    dataset_name = cfg.data.name
    lr = cfg.training.lr
    weight_decay = cfg.training.wd

    if dataset_name == 'ogbg-molhiv':
        aux = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        aux = -1
    optimizers = {
        'zinc12k': torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay),
        'zincfull': torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay),

        'ogbg-molesol': torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay),
        'ogbg-molbace': torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay),
        'ogbg-molhiv': ASAM(aux, model, rho=0.5),

        'alchemy': torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay),

        'Peptides-func': torch.optim.AdamW(model.parameters(), lr=lr,
                                           betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay),
        'Peptides-struc': torch.optim.AdamW(model.parameters(), lr=lr,
                                            betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)

    }
    if dataset_name not in optimizers:
        raise ValueError(
            f"No Optimizer available for the dataset: {dataset_name}")

    return optimizers[dataset_name]


class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()

# --------------------------------- scheduler -------------------------------- #


def get_sched_func(cfg, optim, warmup_epochs=10):
    dataset_name = cfg.data.name
    total_epochs = cfg.training.epochs
    max_lr = cfg.training.lr
    if dataset_name == 'ogbg-molhiv':
        return None

    if dataset_name == 'Peptides-func' or dataset_name == 'Peptides-struc':
        scheduler_peptides = WarmupCosineAnnealingLR(
            optimizer=optim, warmup_epochs=warmup_epochs, total_epochs=total_epochs, max_lr=max_lr)
    else:
        scheduler_peptides = -1

    sched = {
        'zinc12k': torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=cfg.training.patience,
                                                              verbose=True),
        'zincfull': torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=cfg.training.patience,
                                                               verbose=True),
        'ogbg-molesol': torch.optim.lr_scheduler.ConstantLR(optim, total_iters=0, factor=1),
        'ogbg-molbace': torch.optim.lr_scheduler.ConstantLR(optim, total_iters=0, factor=1),

        'alchemy': torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=cfg.training.patience,
                                                              min_lr=0.0000001),

        'Peptides-func': scheduler_peptides,
        'Peptides-struc': scheduler_peptides,
    }

    if dataset_name not in sched:
        raise ValueError(
            f"No Scheduler available for the dataset: {dataset_name}")

    return sched[dataset_name]


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, max_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.initial_lr = 0
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.initial_lr + \
                (self.max_lr - self.initial_lr) * \
                self.last_epoch / self.warmup_epochs
        else:
            cosine_epoch = torch.tensor(
                float(self.last_epoch - self.warmup_epochs))
            cosine_epochs = torch.tensor(
                float(self.total_epochs - self.warmup_epochs))
            lr = self.max_lr * \
                (1 + torch.cos(torch.pi * cosine_epoch / cosine_epochs)) / 2
            lr = lr.item()
        return [lr for _ in self.optimizer.param_groups]

# --------------------------------- evaluator -------------------------------- #


def get_evaluator(cfg):
    dataset_name = cfg.data.name
    evaluators = {
        'zinc12k': ZincLEvaluator(),
        'zincfull': ZincLEvaluator(),
        'ogbg-molesol': Evaluator(name='ogbg-molesol'),
        'ogbg-moltox21': Evaluator(name='ogbg-moltox21'),
        'ogbg-molbace': Evaluator(name='ogbg-molbace'),
        'ogbg-moltoxcast': Evaluator(name='ogbg-moltoxcast'),
        'ogbg-molhiv': Evaluator(name='ogbg-molhiv'),
        'alchemy': ZincLEvaluator(),
        'Peptides-func': AP_eveluator(),
        'Peptides-struc': ZincLEvaluator(),
    }

    if dataset_name not in evaluators:
        raise ValueError(
            f"No loss function available for the dataset: {dataset_name}")

    return evaluators[dataset_name]


class AP_eveluator():
    def eval(self, input_dict):
        '''
        compute Average Precision (AP) averaged across tasks
        '''
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
        ap_list = []
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:, i] == y_true[:, i]
                ap = average_precision_score(y_true[is_labeled, i],
                                             y_pred[is_labeled, i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError(
                'No positively labeled data available. Cannot compute Average Precision.')
        eval_value = sum(ap_list) / len(ap_list)
        AP_dict = {
            'AP': eval_value
        }
        return AP_dict


class ZincLEvaluator(nn.L1Loss):
    def forward(self, input_dict):
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        return super().forward(y_pred, y_true)

    def eval(self, input_dict):
        L1_val = self.forward(input_dict)
        L1_val_dict = {
            'L1loss': L1_val.item()
        }
        return L1_val_dict


# --------------------------------- training -------------------------------- #

def train_loop_ASAM(model, loader, critn, optim, epoch, device, task='regression'):
    model.train()
    loss_list = []
    pbar = tqdm(loader, total=len(loader))
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            # ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            pred = model(copy.deepcopy(batch))
            y = batch.y.view(pred.shape).to(
                torch.float32) if pred.size(-1) == 1 else batch.y
            loss = critn(pred.to(torch.float32)[is_labeled], y[is_labeled])
            loss.backward(retain_graph=True)
            optim.ascent_step()
            # Descent
            pred_ = model(batch)  # already encoded the atom
            loss = critn(pred_.to(torch.float32)[is_labeled], y[is_labeled])
            loss.backward()
            optim.descent_step()
            loss_list.append(loss.item())
            pbar.set_description(
                f"Epoch {epoch} Train Step {i}: Loss = {loss.item()}")
    
    return loss_list


def train_loop(model, loader, critn, optim, epoch, device, task='regression'):
    model.train()
    loss_list = []
    pbar = tqdm(loader, total=len(loader))
    for i, batch in enumerate(pbar):
        batch = batch.to(device)
        optim.zero_grad()
        if task == 'classification':
            is_labeled = batch.y == batch.y
            pred = model(batch)  # pred is 128 x 12

            labeled_y = batch.y.to(torch.float32)[is_labeled]
            labeled_pred = pred.to(torch.float32)[is_labeled]
            
            labeled_y = labeled_y.reshape(-1)
            labeled_pred = labeled_pred.reshape(-1)

            assert labeled_y.shape == labeled_pred.shape
            loss = critn(labeled_pred, labeled_y)
        elif task == 'regression':
            pred = model(batch).view(batch.y.shape)
            loss = critn(pred, batch.y)
        else:
            raise ValueError(
                f"Invalid task type: {task}. Expected 'regression' or 'classification'.")

        loss.backward()
        optim.step()

        loss_list.append(loss.item())
        pbar.set_description(
            f"Epoch {epoch} Train Step {i}: Loss = {loss.item()}")

    return loss_list


def eval_loop(model, loader, eval, device, average_over=1, task='regression'):
    model.eval()
    # Warning: for average_over > 0 this works only if test/val dataloader doesn't shuffle!!!
    input_dict_for_votes = []
    for vote in range(average_over):
        pbar = tqdm(loader, total=len(loader),
                    desc=f"Vote {vote + 1} out of {average_over} votes")
        pred, true = [], []
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            with torch.no_grad():
                if task == 'classification':
                    model_pred = model(batch)
                    true.append(batch.y.view(model_pred.shape).detach().cpu())
                    pred.append(model_pred.detach().cpu())
                elif task == 'regression':
                    true.append(batch.y)
                    pred.append(model(batch).view(batch.y.shape))
                else:
                    raise ValueError(
                        f"Invalid task type: {task}. Expected 'regression' or 'classification'.")

        input_dict = {
            "y_true": torch.cat(true, dim=0),
            "y_pred": torch.cat(pred, dim=0)
        }

        input_dict_for_votes.append(input_dict)
    average_votes_dict = average_dicts(*input_dict_for_votes)
    input_dict = average_votes_dict
    metric = eval.eval(input_dict)
    # Warning: assuming 'metric' is a dictionary with 1 single key!
    metric = list(metric.values())[0]
    return metric


def eval_loop_peptides(model, loaders, eval, device, average_over=1, task='regression', eval_type='val'):
    assert eval_type in [
        'val', 'test'], f"The variable eval_type has to be either 'val' or 'test'"
    model.eval()
    # Warning: for average_over > 0 this works only if test/val dataloader doesn't shuffle!!!
    input_dict_for_votes = []
    for vote in range(average_over):
        loader = loaders[vote][eval_type]
        pbar = tqdm(loader, total=len(loader),
                    desc=f"Vote {vote + 1} out of {average_over} votes")
        pred, true = [], []
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            with torch.no_grad():
                if task == 'classification':
                    model_pred = model(batch)
                    true.append(batch.y.view(
                        model_pred.shape).detach().cpu())
                    pred.append(model_pred.detach().cpu())
                elif task == 'regression':
                    true.append(batch.y)
                    pred.append(model(batch).view(batch.y.shape))
                else:
                    raise ValueError(
                        f"Invalid task type: {task}. Expected 'regression' or 'classification'.")

        input_dict = {
            "y_true": torch.cat(true, dim=0),
            "y_pred": torch.cat(pred, dim=0)
        }
        input_dict_for_votes.append(input_dict)
    average_votes_dict = average_dicts(*input_dict_for_votes)
    input_dict = average_votes_dict
    metric = eval.eval(input_dict)
    # Warning: assuming 'metric' is a dictionary with 1 single key!
    metric = list(metric.values())[0]
    return metric


def sched_step(cfg, sched, val_metric):
    if cfg.data.name == "Peptides-struc" or cfg.data.name == "Peptides-func":
        sched.step()
        return
    try:
        sched.step(val_metric)
    except (KeyError, IndexError, AttributeError) as e:
        logging.info(f"No schedular is found - so no stepping.")

# --------------------------------- training - helpers -------------------------------- #


def average_dicts(*input_dicts):
    # Check if there is at least one dictionary
    if len(input_dicts) == 0:
        raise ValueError("No dictionaries provided.")

    # If only one dictionary is provided, return it as is
    if len(input_dicts) == 1:
        return input_dicts[0]

    # Check if all dictionaries have the same keys
    keys = set(input_dicts[0].keys())
    if not all(keys == set(d.keys()) for d in input_dicts):
        raise ValueError("All dictionaries must have the same keys.")

    averaged_dict = {}

    num_dicts = len(input_dicts)
    for key in keys:
        # Sum the tensors from all dictionaries for the same key
        total_tensor = sum(d[key] for d in input_dicts)

        # Calculate the average
        averaged_tensor = total_tensor / num_dicts
        averaged_dict[key] = averaged_tensor

    return averaged_dict
