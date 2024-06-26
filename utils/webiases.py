import wandb
import logging

def set_wandb(cfg):
    model_name = cfg.model.model_name
    num_layer = cfg.model.num_layer
    dim_embed = cfg.model.dim_embed

    lr = cfg.training.lr
    wd = cfg.training.wd
    epochs = cfg.training.epochs
    subgraph_dropout_prob = 1 - cfg.data.sampling.keep_subgraph_prob
    bs = cfg.data.bs
    data_name = cfg.data.name
    project_name = cfg.wandb.project_name
    seed = cfg.general.seed
    
    model_dropout = cfg.model.dropout

    tag = f"Model_{model_name}||Num_layers_{num_layer}||Bs_{bs}||dim_embed_{dim_embed}||SEED_{seed}||LR_{lr}||WD_{wd}||Epochs_{epochs}||Subgraph_Dropout_p_{subgraph_dropout_prob}||Model_Dropout_p_{model_dropout}||Dataset_{data_name}"
    logging.info(f"{tag}")

    wandb.init(settings=wandb.Settings(
        start_method='thread'), project=project_name, name=tag, config=cfg)