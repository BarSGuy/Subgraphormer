import logging
import wandb
import numpy as np

# --------------------------------- logging code -------------------------------- #

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Initialized logger")


# --------------------------------- logging wandb -------------------------------- #
    
def initialize_best_metrics(goal = 'minimize'):
    assert goal in ["minimize", "maximize"], "Invalid goal: must be either 'minimize' or 'maximize'"
    return {
        "val_loss": float('inf') if goal == "minimize" else float('-inf'),
        "test_loss": float('inf') if goal == "minimize" else float('-inf'),
        "epoch": -1
    }

def update_best_metrics(best_metrics, val_metric, test_metric, epoch, goal = 'minimize'):
    assert goal in ["minimize", "maximize"], "Invalid goal: must be either 'minimize' or 'maximize'"
    if (goal == "minimize" and val_metric < best_metrics["val_loss"]) or \
       (goal == "maximize" and val_metric > best_metrics["val_loss"]):
        best_metrics.update({
            "val_loss": val_metric,
            "test_loss": test_metric,
            "epoch": epoch
        })
    return best_metrics

def log_wandb(epoch, optim, loss_list, val_metric, test_metric, best_metrics):
    try:
        lr = optim.param_groups[0]['lr']
    except (KeyError, IndexError, AttributeError) as e:
        logging.info(f"An error occurred while accessing the learning rate: {e}")
        lr = -1
    
    wandb.log({
        "Epoch": epoch,
        "Train Loss": np.mean(loss_list),
        "Val Loss": val_metric,
        "Test Loss": test_metric,
        "Learning Rate": lr,  # Log the learning rate
        # unpack best metrics into the lognv
        **{f"best_{key}": value for key, value in best_metrics.items()}
    })

def set_posfix(optim, loss_list, val_metric, test_metric, pbar):
    try:
        lr = optim.param_groups[0]['lr']
    except (KeyError, IndexError, AttributeError) as e:
        logging.info(f"An error occurred while accessing the learning rate: {e}")
        lr = -1
    pbar.set_postfix({
        "lr": lr,
        "loss": np.mean(loss_list),
        "val": val_metric,
        "test": test_metric
    })
