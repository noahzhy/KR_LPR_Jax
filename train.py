import os, sys, random, time, glob, math
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

import jax
if sys.platform == "darwin":
    jax.config.update('jax_platform_name', 'cpu')

print(jax.devices())

import yaml
import optax
import jax.numpy as jnp

sys.path.append("./utils")
from model.loss import *
from model.model import TinyLPR
from model.dataloader import get_data
from fit import lr_schedule, fit, TrainState, load_ckpt
from utils import batch_ctc_greedy_decoder, batch_remove_blank


cfg = yaml.safe_load(open("config.yaml"))
print(cfg)

val_ds, _ = get_data(**cfg["val"])
train_ds, train_len = get_data(**cfg["train"])

lr_fn = lr_schedule(cfg["lr"], train_len, cfg["epochs"], cfg["warmup"])


@jax.jit
def loss_fn(pred, target, step=None):
    pred_mask, pred_feat, pred_ctc = pred
    mask, label = target

    # loss_ctc = focal_ctc_loss(pred_ctc, label, **cfg["focal_ctc_loss"])
    # loss_mask = dice_bce_loss(pred_mask, mask)
    loss_ctc = ctc_loss(pred_ctc, label, **cfg["ctc_loss"]).mean()
    loss_mask = tversky_loss(pred_mask, mask)
    loss_center = center_ctc_loss((pred_feat, pred_ctc), **cfg["center_ctc_loss"])

    loss_center = jax.lax.cond(
        step <= train_len * 20,
        lambda x: jnp.array(.0, dtype=jnp.float32),
        lambda x: loss_center,
        None
    )

    loss = (cfg["focal_ctc_loss"]["weight"] * loss_ctc +
            cfg["dice_bce_loss"]["weight"]  * loss_mask +
            cfg["center_ctc_loss"]["weight"]* loss_center)

    loss_dict = {
        'loss': loss,
        'loss_ctc': loss_ctc,
        'loss_mask': loss_mask,
        'loss_center': loss_center,
    }
    return loss, loss_dict


@jax.jit
def predict(state: TrainState, batch):
    img, (_, label) = batch
    pred_ctc = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats
        }, img, train=False)
    return pred_ctc, label


@jax.jit
def eval_step(state: TrainState, batch):
    pred_ctc, label = predict(state, batch)
    pred = batch_ctc_greedy_decoder(pred_ctc)
    # replace -1 with 0 in label and pred
    pred = jnp.where(pred == -1, 0, pred)
    label = jnp.where(label == -1, 0, label)
    ans = batch_array_comparison(pred, label, size=cfg["max_len"]+1)
    return jnp.mean(ans)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jnp.zeros((1, *cfg["img_size"], 1))

    model = TinyLPR(**cfg["model"])
    var = model.init(key, x, train=True)
    params = var['params']
    batch_stats = var['batch_stats']

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.inject_hyperparams(optax.nadam)(lr_fn),
    )

    state = load_ckpt(state, cfg["ckpt"])

    fit(state, train_ds, val_ds,
        loss_fn=loss_fn,
        eval_step=eval_step,
        num_epochs=cfg["epochs"],
        eval_freq=cfg["eval_freq"],
        log_name="tinyLPR",
        hparams=cfg,
    )
