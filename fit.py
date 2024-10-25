import os, time
from typing import Any
from functools import partial
from pathlib import Path
from flax import nnx

from tqdm import tqdm
import optax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
# from flax.training import train_state, orbax_utils
import tensorboardX as tbx


def banner_message(message):
    if isinstance(message, str):
        message = [message]
    elif not isinstance(message, list) or not all(isinstance(m, str) for m in message):
        raise ValueError("message should be a string or a list of strings.")

    msg_len = max(46, max(len(msg) for msg in message))

    # Top border
    print("\33[1;32m╔═{:═<{width}}═╗".format("", width=msg_len))
    # Message lines
    for msg in message:
        print("║ {:^{width}} ║".format(msg, width=msg_len))
    # Bottom border
    print("╚═{:═<{width}}═╝\33[0m".format("", width=msg_len))


def lr_schedule(lr, steps_per_epoch, epochs=100, warmup=5):
    return optax.warmup_cosine_decay_schedule(
        init_value=lr / 10,
        peak_value=lr,
        end_value=lr / 100,
        warmup_steps=steps_per_epoch * warmup,
        decay_steps=steps_per_epoch * (epochs - warmup),
    )


@nnx.jit
def eval_step(model, batch):
    x, y = batch
    logits = model(x)
    acc = jnp.equal(jnp.argmax(logits, -1), y).mean()
    return acc


@nnx.jit
def loss_fn(logits, labels, epoch=None):
    loss = optax.softmax_cross_entropy(
        logits=logits,
        labels=jax.nn.one_hot(labels, 10),
    ).mean()
    return loss, {'loss': loss}


@partial(nnx.jit, static_argnums=(3,))
def train_step(model, optimizer: nnx.Optimizer, batch, loss_fn, epoch):
    x, y = batch

    def compute_loss(model):
        logits = model(x)
        return loss_fn(logits, y, epoch)

    grad_fn = nnx.value_and_grad(compute_loss, has_aux=True)
    (loss, loss_dict), grads = grad_fn(model)
    optimizer.update(grads)
    return loss_dict


def load_ckpt(model, ckpt_dir, epoch=None):
    if ckpt_dir is None or not os.path.exists(ckpt_dir):
        banner_message(["No checkpoint was loaded", "Training from scratch"])
        return model

    ckpt_dir = os.path.abspath(ckpt_dir)
    epoch = epoch or max(int(f) for f in os.listdir(ckpt_dir) if f.isdigit())

    checkpointer = ocp.StandardCheckpointer()
    graphdef, abstract_state = nnx.split(model)

    ckpt_path = os.path.join(ckpt_dir, str(epoch))
    state_restored = checkpointer.restore(ckpt_path, abstract_state)
    model = nnx.merge(graphdef, state_restored)
    return model


def fit(model,
        train_ds, test_ds,
        optimizer:nnx.Optimizer,
        loss_fn=loss_fn,
        num_epochs=100,
        eval_step=None,
        eval_freq=1,
        log_name='default',
        hparams=None,
    ):
    # logging
    banner_message(["Start training", "Device > {}".format(", ".join([str(i) for i in jax.devices()]))])

    Path("checkpoints").mkdir(exist_ok=True, parents=True)
    ckpt_path = os.path.abspath("checkpoints")
    checkpointer = ocp.StandardCheckpointer()

    writer = tbx.SummaryWriter("logs/{}_{}".format(log_name, time.strftime("%m%d%H%M%S")))
    best_acc = .0
    # start training
    for epoch in range(1, num_epochs + 1):
        model.train()
        pbar = tqdm(train_ds)
        for batch in pbar:
            ## if batch is not from tfds.as_numpy, convert it to numpy
            batch = jax.tree_map(lambda x: x._numpy(), batch)
            loss_dict = train_step(model, optimizer, batch, loss_fn, epoch)
            # lr = 0.1
            pbar.set_description(f'Epoch {epoch:3d}, loss: {loss_dict["loss"]:.4f}')

            steps = optimizer.step.value
            if steps % 10 == 0 or steps == 1:
                # writer.add_scalar('train/learning_rate', lr, steps)
                for k, v in loss_dict.items():
                    writer.add_scalar(f'train/{k}', v, steps)

            writer.flush()

        if eval_step is None:
            _, state = nnx.split(model)
            checkpointer.save(os.path.join(ckpt_path, str(epoch)), state, force=True)

        elif epoch % eval_freq == 0:
            acc = []
            model.eval()
            for batch in test_ds:
                ## if batch is not from tfds.as_numpy, convert it to numpy
                batch = jax.tree_map(lambda x: x._numpy(), batch)
                a = eval_step(model, batch)
                acc.append(a)

            acc = 0 if len(acc) == 0 else jnp.stack(acc).mean()
            pbar.write(f'Epoch {epoch:3d}, test acc: {acc:.6f}')
            writer.add_scalar('test/accuracy', acc, epoch)

            if acc > best_acc:
                _, state = nnx.split(model)
                checkpointer.save(os.path.join(ckpt_path, str(epoch)), state, force=True)
                best_acc = acc

    banner_message(["Training finished", f"Best test acc: {best_acc:.6f}"])
    if hparams is not None:
        writer.add_hparams(hparams, {'metric/accuracy': best_acc}, name='hparam')
    writer.close()


banner_message(["Device > {}".format(", ".join([str(i) for i in jax.devices()]))])


if __name__ == "__main__":
    from model import Model
    import tensorflow_datasets as tfds


    def get_train_batches(batch_size=256):
        ds = tfds.load(name='mnist', split='train', as_supervised=True, shuffle_files=True)
        ds = ds.batch(batch_size).prefetch(1)
        # return tfds.as_numpy(ds)
        return ds # debug for some reason, tfds.as_numpy(ds) is not working


    def get_test_batches(batch_size=256):
        ds = tfds.load(name='mnist', split='test', as_supervised=True, shuffle_files=False)
        ds = ds.batch(batch_size).prefetch(1)
        # return tfds.as_numpy(ds)
        return ds


    config = {
        'lr': 5e-3,
        'batch_size': 128,
        'num_epochs': 10,
        'warmup': 3,
    }

    train_ds = get_train_batches(batch_size=config['batch_size'])
    test_ds = get_test_batches(batch_size=config['batch_size'])
    lr_fn = lr_schedule(lr=config['lr'], steps_per_epoch=len(train_ds), epochs=config['num_epochs'], warmup=config['warmup'])

    key = nnx.Rngs(0)
    model = Model(key)
    x = jnp.ones((1, 28, 28, 1))

    optimizer = nnx.Optimizer(model, optax.nadam(lr_fn))

    fit(model, train_ds, test_ds,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_step=eval_step,
        eval_freq=1,
        num_epochs=config['num_epochs'],
        hparams=config,
        log_name='mnist')

    model = load_ckpt(model, "./checkpoints")

    acc = []
    model.eval()
    for batch in test_ds:
        batch = jax.tree_map(lambda x: x._numpy(), batch)
        a = eval_step(model, batch)
        acc.append(a)
    acc = jnp.stack(acc).mean()

    print("Eval Accuracy: {:.6f}".format(acc))
