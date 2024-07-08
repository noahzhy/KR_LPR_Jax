import os, time
from typing import Any

import jax
import optax
from tqdm import tqdm
import jax.numpy as jnp
from flax.training import train_state
from flax.training import checkpoints
import tensorboardX as tbx


def banner_message(message):
    print("\33[32m=\33[0m" * 60)
    if isinstance(message, str):
        print("\33[32m== {:^54} ==\33[0m".format(message))
    elif isinstance(message, list):
        for msg in message:
            print("\33[32m== {:^54} ==\33[0m".format(str(msg)))
    else:
        raise ValueError("message should be str or list of str.")
    print("\33[32m=\33[0m" * 60)


def lr_schedule(lr, steps_per_epoch, epochs=100, warmup=5):
    return optax.warmup_cosine_decay_schedule(
        init_value=lr / 10,
        peak_value=lr,
        end_value=1e-5,
        warmup_steps=steps_per_epoch * warmup,
        decay_steps=steps_per_epoch * (epochs - warmup),
    )
    # return optax.cosine_onecycle_schedule(
    #     peak_value=lr,
    #     transition_steps=steps_per_epoch * epochs,
    #     pct_start=0.2,
    # )


class TrainState(train_state.TrainState):
    batch_stats: Any


@jax.jit
def train_step(state: TrainState, batch, opt_state):
    x, y = batch
    def loss_fn(params):
        logits, updates = state.apply_fn({
            'params': params,
            'batch_stats': state.batch_stats
        }, x, train=True, mutable=['batch_stats'])
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, 10)).mean()
        loss_dict = {'loss': loss}
        return loss, (loss_dict, updates)

    (_, (loss_dict, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=updates['batch_stats'])
    _, opt_state = state.tx.update(grads, opt_state)
    return state, loss_dict, opt_state


@jax.jit
def eval_step(state: TrainState, batch):
    x, y = batch
    logits = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats
        }, x, train=False)
    acc = jnp.equal(jnp.argmax(logits, -1), y).mean()
    return state, acc


def load_ckpt(state, ckpt_dir, step=None):
    if ckpt_dir is None or not os.path.exists(ckpt_dir):
        banner_message("No checkpoint was loaded. Training from scratch.")
        return state
    
    banner_message("Loading ckpt from {}".format(ckpt_dir))

    return checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=state,
        step=step,
    )


def fit(state, train_ds, test_ds,
        train_step=train_step, eval_step=eval_step,
        num_epochs=100, log_name='default', eval_freq=1,
    ):
    banner_message(["Start training", "Device: {}".format(jax.devices()[0])])
    timestamp = time.strftime("%Y%m%d%H%M%S")
    writer = tbx.SummaryWriter("logs/{}_{}".format(log_name, timestamp))
    opt_state = state.tx.init(state.params)
    best_acc = .0
    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(train_ds)
        for batch in pbar:
            state, loss_dict, opt_state = train_step(state, batch, opt_state)
            lr = opt_state.hyperparams['learning_rate']
            pbar.set_description(f'Epoch {epoch:3d}, lr: {lr:.7f}, loss: {loss_dict["loss"]:.4f}')

            if state.step % 100 == 0:
                writer.add_scalar('train/lr', lr, state.step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f'train/{k}', v, state.step)

        if epoch % eval_freq == 0:
            acc = []
            for batch in test_ds:
                state, a = eval_step(state, batch)
                acc.append(a)
            acc = jnp.stack(acc).mean()

            pbar.write(f'Epoch {epoch:3d}, test acc: {acc:.4f}')
            writer.add_scalar('test/acc', acc, epoch)
            writer.flush()

            if acc > best_acc:
                best_acc = acc
                ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
                checkpoints.save_checkpoint(
                    ckpt_dir=ckpt_path,
                    target=state,
                    step=epoch,
                    overwrite=True,
                    keep=1,
                )

    banner_message(["Training finished.", f"Best test acc: {best_acc:.6f}"])
    writer.close()


if __name__ == "__main__":
    from model import Model
    import tensorflow_datasets as tfds


    def get_train_batches(batch_size=256):
        ds = tfds.load(name='mnist', split='train', as_supervised=True, shuffle_files=True)
        ds = ds.batch(batch_size).prefetch(1)
        return tfds.as_numpy(ds)


    def get_test_batches(batch_size=256):
        ds = tfds.load(name='mnist', split='test', as_supervised=True, shuffle_files=False)
        ds = ds.batch(batch_size).prefetch(1)
        return tfds.as_numpy(ds)


    train_ds, test_ds = get_train_batches(), get_test_batches()
    lr_fn = lr_schedule(lr=3e-3, steps_per_epoch=len(train_ds), epochs=10, warmup=1)

    key = jax.random.PRNGKey(0)
    model = Model()
    x = jnp.ones((1, 28, 28, 1))
    var = model.init(key, x, train=True)
    params = var['params']
    batch_stats = var['batch_stats']

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.inject_hyperparams(optax.adam)(lr_fn),
    )

    import time
    start = time.perf_counter()
    fit(state, train_ds, test_ds, num_epochs=10, log_name='mnist')
    print("Elapsed time: {} ms".format((time.perf_counter() - start) * 1000))

    state = load_ckpt(state, "checkpoints")

    acc = []
    for batch in test_ds:
        _, a = eval_step(state, batch)
        acc.append(a)
    acc = jnp.stack(acc).mean()
    print(f'Test acc: {acc:.4f}')
