import os
import json
import numpy as onp

import dill
from comet_ml import Experiment
import jax
from jax import numpy as jnp, random
from jax.random import split, PRNGKey
import haiku as hk
import optax as ox

from trainer.supervised import ModelWrapper
from trainer.meta import CLWrapper
from losses import mean_xe_and_acc_dict

from data import augment
from data.sampling import BatchSampler, MRCLDatasetLoader
from datasets import get_dataset
from train_mrcl_rebuilt import (
    Subset,
    # get_dataset,
    make_test_iterators,
    OMLConvnet,
    ANMLNet,
)
from tqdm import tqdm

import logging
from logging import handlers
from torch.utils.tensorboard import SummaryWriter

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("experiment")


def main(args):
    rng = PRNGKey(args.seed)
    rng, rng_inner_loader, rng_outer_loader = split(rng, 3)

    args.exp_name = args.exp_name + "_" + str(args.seed)
    logdir = os.path.join(args.logdir, args.exp_name)
    if logdir:
        os.makedirs(logdir)
        with open(os.path.join(logdir, "args.json"), "w") as f:
            json.dump(vars(args), f)

    fh = logging.FileHandler(os.path.join(logdir, "log.txt"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        # logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
        logging.Formatter("%(levelname)-8s %(message)s")
    )
    logger.addHandler(fh)

    ch = logging.handlers.logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        # logging.Formatter('rank:' + str(args['rank']) + ' ' + name + ' %(levelname)-8s %(message)s'))
        logging.Formatter("%(levelname)-8s %(message)s")
    )
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.info(str(vars(args)))

    if args.logcomet:
        comet_api_key = os.environ["COMET_API_KEY"]
    else:
        comet_api_key = os.environ.get("COMET_API_KEY", default="")
    comet = Experiment(
        api_key=comet_api_key,
        workspace="the-thesis",
        project_name=args.comet_project,
        disabled=not args.logcomet,
    )
    comet.set_name(args.exp_name)
    comet.log_parameters(vars(args))

    train_dataset = get_dataset(
        "omniglot",
        args.train_split,
        train=True,
        all=True,
        image_size=args.image_size,
        data_dir=args.data_dir,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")

    if args.train_split == "train":
        num_tasks = 664
    elif args.train_split == "train+val":
        num_tasks = 964

    test_dataset = get_dataset(
        "omniglot",
        f"{args.test_split}_train",
        train=args.test_split == "val",
        image_size=args.image_size,
        all=True,
        data_dir=args.data_dir,
    )
    # test_test_dataset = get_dataset(
    #     "omniglot",
    #     f"{args.test_split}_test",
    #     train=args.test_split == "val",
    #     image_size=args.image_size,
    # )
    logger.info(f"Test dataset size: {len(test_dataset)}")
    # logger.info(f"Test test dataset size: {len(test_test_dataset)}")

    bsz = args.batch_size
    rng, rng_data1, rng_data2, rng_slow, rng_fast = split(rng, 5)

    MODEL = args.model.upper()
    BSZ = args.batch_size

    if MODEL == "ANML":
        _model = ANMLNet
    elif MODEL == "OML":
        _model = OMLConvnet
    else:
        _model = None

    train_iterator = BatchSampler(
        rng_data2,
        train_dataset,
        BSZ,
        shuffle=True,
        keep_last=False,
        dataset_is_array=True,
    )

    model = hk.transform_with_state(
        lambda x, mask=None, phase="all", training=None: _model(1000, 9)(
            x,
            mask=mask,
            phase=phase,
            training=training,
        )
    )

    dummy_input, dummy_targets = next(iter(train_iterator))

    if args.normalize_input:
        normalize_fn = train_dataset.normalize
    else:
        normalize_fn = lambda x: x
    params, state = model.init(rng_slow, normalize_fn(dummy_input / 255), phase="all")

    lr = jnp.array(args.lr)
    # sch_dict = {
    #     600: 0.1,
    #     800: 0.1,
    # }
    sch_dict = args.schedule

    trainer = ModelWrapper(
        model.apply,
        params,
        state,
    )

    optimizer = ox.chain(
        ox.additive_weight_decay(5e-4),
        ox.trace(decay=0.9, nesterov=False),
    )

    def get_step_lr(step_num, lr, sch_dict):
        return ox.piecewise_constant_schedule(lr, sch_dict)(step_num)

    def schedule(step_num, updates, lr, sch_dict):
        return ox.scale(get_step_lr(step_num, lr, sch_dict)).update(updates, None)[0]

    # train_preprocess_fn = lambda rng, inputs: normalize_fn(inputs / 255)

    if args.augment:
        train_preprocess_fn = lambda rng, inputs: normalize_fn(
            augment(rng, inputs / 255, out_size=args.image_size)
        )
    else:
        train_preprocess_fn = lambda rng, inputs: normalize_fn(inputs / 255)

    trainer.init_opt_state(optimizer).set_step_fn(
        jax.jit(
            trainer.make_step_fn(
                optimizer,
                jax.partial(schedule, lr=-lr, sch_dict=sch_dict),
                mean_xe_and_acc_dict,
                train_preprocess_fn,
            )
        )
    )

    test_optimizer = ox.adam(0)

    def test_inner_opt_update_fn(lr, updates, state, params):
        inner_opt = ox.adam(lr)
        return inner_opt.update(updates, state, params)

    def reset_fast_params_fn(w_make_fn, rng, spt_classes, params, all=False):
        if MODEL == "ANML":
            w_name = "anml_net/classifier/linear"
        else:
            w_name = "oml_convnet/linear"
        params = hk.data_structures.to_mutable_dict(params)
        w = params[w_name]["w"]
        if all:
            params[w_name]["w"] = w_make_fn(dtype=w.dtype)(
                rng,
                w.shape,
            )
        else:
            params[w_name]["w"] = jax.ops.index_update(
                w,
                jax.ops.index[:, spt_classes],
                w_make_fn(dtype=w.dtype)(rng, (w.shape[0], spt_classes.shape[0])),
            )
        return hk.data_structures.to_immutable_dict(params)

    test_preprocess_fn = lambda inputs: normalize_fn(inputs / 255)

    if MODEL == "ANML":
        val_slow_apply = jax.partial(model.apply, phase="test_encoder")
        val_fast_apply = jax.partial(model.apply, phase="test_adaptation")
    elif MODEL == "OML":
        val_slow_apply = jax.partial(model.apply, phase="encoder")
        val_fast_apply = jax.partial(model.apply, phase="adaptation")

    test_sup_model = CLWrapper(
        val_slow_apply=val_slow_apply,
        val_fast_apply=val_fast_apply,
        training=False,
        loss_fn=mean_xe_and_acc_dict,
        test_init_inner_opt_state_fn=test_optimizer.init,
        test_inner_opt_update_fn=test_inner_opt_update_fn,
        preprocess_test_fn=jax.jit(lambda x: normalize_fn(x / 255)),
    )

    reset_fn = jax.nn.initializers.he_normal

    if MODEL == "ANML":
        w_name = "classifier"
    elif MODEL == "OML":
        w_name = "oml_convnet/linear"

    def get_test_slow_params(params):
        return hk.data_structures.partition(
            lambda module_name, name, value: w_name in module_name,
            params,
        )[1]

    def get_test_fast_params(
        rng,
        params,
        zero_weight=True,
        keep_bias=False,
    ):
        tmp_fast_params = hk.data_structures.partition(
            lambda module_name, name, value: w_name in module_name,
            params,
        )[0]

        weights, biases = hk.data_structures.partition(
            lambda module_name, name, value: name == "w", tmp_fast_params
        )
        tmp_fast_params = hk.data_structures.merge(
            jax.tree_map(lambda t: jnp.zeros_like(t), weights)
            if zero_weight
            else jax.tree_map(lambda t: reset_fn(dtype=t.dtype)(rng, t.shape), weights),
            biases if keep_bias else jax.tree_map(lambda t: jnp.zeros_like(t), biases),
        )
        return tmp_fast_params

    if args.test_split == "test":
        total_num_test_classes = 659
        test_traj_length = 600
    elif args.test_split == "val":
        total_num_test_classes = 300
        test_traj_length = 300

    logger.info(
        f"Test total num classes: {total_num_test_classes}, test traj length: {test_traj_length}"
    )

    test_classes_arange = jnp.arange(total_num_test_classes)
    # test_traj_length = test_traj_length
    test_train_iterators, test_test_iterators = make_test_iterators(
        PRNGKey(1),
        test_dataset,
        test_dataset.targets,
        5,
        test_classes_arange,
        test_traj_length,
        sort=args.sorted_test,
        shuffle=args.shuffle_test,
    )

    # print(trainer.params)
    # print(get_test_slow_params(trainer.params))
    # print(get_test_fast_params(None, trainer.params, zero_weight=True, keep_bias=False))

    test_sup_model.test(
        test_train_iterators[0],
        test_test_iterators[0],
        inner_lr=0.00001,
        slow_params=get_test_slow_params(trainer.params),
        fast_params=get_test_fast_params(
            PRNGKey(0), trainer.params, zero_weight=True, keep_bias=False
        ),
        slow_state=trainer.state,
        fast_state=trainer.state,
    )
    lr_sweep = args.lr_sweep

    histories = {
        "train": {
            "loss": [],
            "acc": [],
            "lr": [],
            "epoch": [],
        },
        "val": {
            "loss_train": {lr: [] for lr in lr_sweep},
            "acc_train": {lr: [] for lr in lr_sweep},
            "loss_test": {lr: [] for lr in lr_sweep},
            "acc_test": {lr: [] for lr in lr_sweep},
            "epoch": [],
        },
    }

    best_val_acc = 0
    loss_ema = aux_ema = None
    # iterator = iter(mrcl_loader)
    num_epochs = args.num_epochs
    train_pbar = tqdm(
        range(1, num_epochs + 1),
        ncols=0,
        mininterval=2.5,
    )

    writer = SummaryWriter(os.path.join(logdir, "tensorboard"))

    for epoch in train_pbar:
        loss_ema = aux_ema = None
        for i, (x, y) in enumerate(train_iterator):
            rng, rng_step = split(rng)
            loss, aux = trainer.train_step(epoch, rng_step, x, y)

            if loss_ema is None:
                loss_ema = loss
                aux_ema = aux
            else:
                (loss_ema, aux_ema) = jax.tree_multimap(
                    lambda ema, x: ema * 0.9 + x * 0.1,
                    (loss_ema, aux_ema),
                    (loss, aux),
                )

            if (i % 100) == 0:
                train_pbar.set_postfix(
                    loss=loss_ema.item(),
                    acc=aux_ema["acc"].item(),
                    refresh=False,
                )

        loss = loss_ema.item()
        acc = aux_ema["acc"].item()
        train_pbar.set_postfix(
            loss=loss,
            acc=acc,
        )

        histories["train"]["loss"].append(loss)
        histories["train"]["acc"].append(acc)
        histories["train"]["epoch"].append(epoch)
        step_lr = get_step_lr(
            epoch,
            lr=lr,
            sch_dict=sch_dict,
        ).item()
        histories["train"]["lr"].append(step_lr)

        train_metrics = {
            "lr": step_lr,
            "loss": loss,
            "acc": acc,
        }

        for tag, metric in train_metrics.items():
            writer.add_scalar(
                "train/" + tag,
                metric,
                global_step=epoch,
            )

        comet.log_metrics(
            train_metrics,
            prefix="train",
            step=epoch,
        )

        if (
            ((epoch) % args.test_interval) == 0
            or ((epoch) == args.num_epochs)
            or (epoch == 1)
        ):
            histories["val"]["epoch"].append(epoch)
            logger.info("\nTrain metrics")
            logger.info(str(train_metrics))
            logger.info(f"TEST EPOCH {epoch}")

            train_accs = {}
            test_accs = {}
            train_losses = {}
            test_losses = {}

            tmp_slow_params = get_test_slow_params(trainer.params)
            tmp_fast_params = get_test_fast_params(
                None,
                trainer.params,
                zero_weight=True,
                keep_bias=False,
            )

            for inner_lr in lr_sweep:
                train_accs[inner_lr] = []
                test_accs[inner_lr] = []
                train_losses[inner_lr] = []
                test_losses[inner_lr] = []

                for test_train_iterator, test_test_iterator in zip(
                    test_train_iterators, test_test_iterators
                ):
                    (
                        (test_train_loss, test_train_acc),
                        (test_test_loss, test_test_acc),
                    ) = test_sup_model.test(
                        test_train_iterator,
                        test_test_iterator,
                        inner_lr=inner_lr,
                        slow_params=tmp_slow_params,
                        fast_params=tmp_fast_params,
                        slow_state=trainer.state,
                        fast_state=trainer.state,
                    )
                    # print(test_train_loss, test_train_acc, test_test_loss, test_test_acc)
                    train_accs[inner_lr].append(test_train_acc.item())
                    test_accs[inner_lr].append(test_test_acc.item())
                    train_losses[inner_lr].append(test_train_loss.item())
                    test_losses[inner_lr].append(test_test_loss.item())

                histories["val"]["loss_train"][inner_lr].append(train_losses[inner_lr])
                histories["val"]["loss_test"][inner_lr].append(test_losses[inner_lr])
                histories["val"]["acc_train"][inner_lr].append(train_accs[inner_lr])
                histories["val"]["acc_test"][inner_lr].append(test_accs[inner_lr])

                logger.info(f"Test lr: {inner_lr}")
                logger.info("Test train acc: %f" % onp.mean(train_accs[inner_lr]))
                logger.info("Test test acc: %f" % onp.mean(test_accs[inner_lr]))
                logger.info("Test train loss: %f" % onp.mean(train_losses[inner_lr]))
                logger.info("Test test loss: %f" % onp.mean(test_losses[inner_lr]))

            writer.add_scalars(
                "test/loss_train",
                {
                    str(lr): onp.mean(v[-1])
                    for lr, v in histories["val"]["loss_train"].items()
                },
                global_step=epoch,
            )
            writer.add_scalars(
                "test/loss_test",
                {
                    str(lr): onp.mean(v[-1])
                    for lr, v in histories["val"]["loss_test"].items()
                },
                global_step=epoch,
            )
            writer.add_scalars(
                "test/acc_train",
                {
                    str(lr): onp.mean(v[-1])
                    for lr, v in histories["val"]["acc_train"].items()
                },
                global_step=epoch,
            )
            writer.add_scalars(
                "test/acc_test",
                {
                    str(lr): onp.mean(v[-1])
                    for lr, v in histories["val"]["acc_test"].items()
                },
                global_step=epoch,
            )

            test_metrics = {}
            test_metrics.update(
                {
                    f"loss_train_{lr}": onp.mean(v[-1])
                    for lr, v in histories["val"]["loss_train"].items()
                }
            )
            test_metrics.update(
                {
                    f"loss_test_{lr}": onp.mean(v[-1])
                    for lr, v in histories["val"]["loss_test"].items()
                }
            )
            test_metrics.update(
                {
                    f"acc_train_{lr}": onp.mean(v[-1])
                    for lr, v in histories["val"]["acc_train"].items()
                }
            )
            test_metrics.update(
                {
                    f"acc_test_{lr}": onp.mean(v[-1])
                    for lr, v in histories["val"]["acc_test"].items()
                }
            )

            comet.log_metrics(
                test_metrics,
                prefix="test",
                # step=epoch,
                epoch=epoch,
            )

            best_lr = max(lr_sweep, key=lambda k: onp.mean(test_accs[k]))
            val_acc = onp.mean(test_accs[best_lr])
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # logger.info("New best val acc:" + str(best_val_acc))

                logger.info(
                    "\nNEW BEST VAL ACC: "
                    + str(best_val_acc)
                    + " learning rate: "
                    + str(best_lr)
                    + "\n"
                )

                with open(os.path.join(logdir, "histories_best.json"), "w") as f:
                    json.dump(histories, f)
                with open(os.path.join(logdir, "best.dill"), "wb") as f:
                    dill.dump(
                        {
                            "histories": histories,
                            "params": trainer.params,
                            "state": trainer.state,
                            "opt_state": trainer.opt_state,
                            "inner_lr": best_lr,
                        },
                        f,
                    )

        if ((epoch % 200) == 0) or (epoch in list(sch_dict.keys())):
            # if (epoch) in list(sch_dict.keys()) + [200]:
            with open(os.path.join(logdir, f"histories_{epoch}.json"), "w") as f:
                json.dump(histories, f)
            with open(os.path.join(logdir, f"model_{epoch}.dill"), "wb") as f:
                dill.dump(
                    {
                        "histories": histories,
                        "params": trainer.params,
                        "state": trainer.state,
                        "opt_state": trainer.opt_state,
                        "histories": histories,
                        "best_val_acc": best_val_acc,
                    },
                    f,
                )

    with open(os.path.join(logdir, "histories.json"), "w") as f:
        json.dump(histories, f)

    with open(os.path.join(logdir, "model.dill"), "wb") as f:
        dill.dump(
            {
                "histories": histories,
                "params": trainer.params,
                "state": trainer.state,
                "opt_state": trainer.opt_state,
                "histories": histories,
                "best_val_acc": best_val_acc,
            },
            f,
        )


import argparse


class keyvalue(argparse.Action):
    # Constructor calling
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            # split it into key and value
            key, value = value.split("=")
            # assign into dictionary
            getattr(namespace, self.dest)[key] = value


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="../data/from_torch/omniglot")

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_interval", type=int, default=20)
    parser.add_argument("--sorted_test", type=int, default=0, choices=[0, 1])
    parser.add_argument("--shuffle_test", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--augment",
        default=False,
        action="store_true",
    )

    parser.add_argument("--normalize_input", default=False, action="store_true")
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--schedule", nargs="*", action=keyvalue, default={600: 0.1, 800: 0.1})
    parser.add_argument("--model", type=str, choices=["oml", "anml"], required=True)
    parser.add_argument("--image_size", type=int, default=28, choices=[28, 84])
    parser.add_argument(
        "--train_split", type=str, choices=["train", "train+val"], default="train"
    )
    parser.add_argument(
        "--test_split", type=str, choices=["val", "test"], default="val"
    )
    parser.add_argument("--test_zero_weight", type=int, required=True, choices=[0, 1])
    parser.add_argument("--test_keep_bias", type=int, required=True, choices=[0, 1])
    parser.add_argument("--logcomet", default=False, action="store_true")
    parser.add_argument("--comet_project", default="continual_learning2", type=str)

    parser.add_argument(
        "--lr_sweep",
        type=float,
        nargs="+",
        default=[
            #     0.01,
            # 0.0085,
            # 0.005,
            # 0.003,
            # 0.001,
            0.00085,
            0.0005,
            0.0003,
            0.0001,
            0.000085,
            0.00005,
            0.00003,
            0.00001,
        ],
    )

    args = parser.parse_args()
    args.schedule = {int(k): float(v) for k, v in args.schedule.items()}
    return args


if __name__ == "__main__":
    main(parse_args())
