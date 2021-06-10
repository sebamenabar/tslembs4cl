import os
import os.path as osp
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import scipy
import numpy as onp
from easydict import EasyDict as edict

import dill
import json
from tqdm import tqdm

import jax
from jax import numpy as jnp, random
from jax.random import split, PRNGKey

import haiku as hk
import optax as ox

from losses import mean_xe_and_acc_dict
from train_mrcl_rebuilt import (
    ImageDataset,
    Subset,
    get_dataset,
    CLWrapper,
    OMLConvnet,
    ANMLNet,
    BatchSampler,
    sample_trajectory,
)



def main(args):
    print("Arguments:")
    print(vars(args))

    with open(osp.join(args.model_dir, "args.json"), "r") as f:
        metadata = edict(json.load(f))

    print("Metadata:")
    print(metadata)

    with open(osp.join(args.model_dir, args.model_filename), "rb") as f:
        state = dill.load(f)

    if metadata.model == "anml":
        _model = ANMLNet
    elif metadata.model == "oml":
        _model = OMLConvnet

    model = hk.transform_with_state(
        lambda x, mask=None, phase="all", training=None: _model(
            1000, 9, image_size=metadata.image_size
        )(
            x,
            mask=mask,
            phase=phase,
            training=training,
        )
    )
    train_dataset = get_dataset(
        "omniglot", metadata.train_split, train=True, all=True, image_size=metadata.image_size, data_dir=args.data_dir,
    )
    test_dataset = get_dataset(
        "omniglot", "test_train", train=False, all=True, image_size=metadata.image_size, data_dir=args.data_dir,
    )
    
    if metadata.normalize_input:
        normalize_fn = train_dataset.normalize
    else:
        normalize_fn = lambda x: x
    test_preprocess_fn = lambda inputs: normalize_fn(inputs / 255)

    test_optimizer = ox.adam(0)

    def test_inner_opt_update_fn(lr, updates, state, params):
        inner_opt = ox.adam(lr)
        return inner_opt.update(updates, state, params)

    def reset_fast_params_fn(w_make_fn, rng, spt_classes, params, all=False):
        if metadata.model == "anml":
            w_name = "anml_net/classifier/linear"
        elif metadata.model == "oml":
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

    if metadata.model == "anml":
        val_slow_apply = jax.partial(model.apply, phase="test_encoder")
        val_fast_apply = jax.partial(model.apply, phase="test_adaptation")
    elif metadata.model == "oml":
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

    if metadata.model == "anml":
        w_name = "classifier"
    elif metadata.model == "oml":
        w_name = "oml_convnet/linear"

    reset_fn = jax.nn.initializers.he_normal
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

    tmp_slow_params = jax.tree_map(
        jax.device_put, get_test_slow_params(state["params"]),
    )
    tmp_fast_params = jax.tree_map(
        jax.device_put, get_test_fast_params(None, state["params"], zero_weight=args.zero_weight, keep_bias=args.keep_bias),
    )

    train_train_search_results_all = {
        t: {lr: [] for lr in args.lr_sweep}
        for t in args.trajs
        # t: [] for t in trajs
    }
    train_train_search_results_bests = {t: [] for t in args.trajs}

    trajs_pbar = tqdm(args.trajs)
    runs_pbar = tqdm(range(args.num_runs_search))
    lrs_pbar = tqdm(args.lr_sweep)

    for traj_length in trajs_pbar:
        runs_pbar.reset()
        runs_pbar.n = 0
        runs_pbar.refresh()
        for n in range(args.num_runs_search):
            rng, random_state = split(PRNGKey(traj_length + n))

            test_train_dataset, _ = sample_trajectory(
                random_state,
                test_dataset,
                test_dataset.targets,
                traj_length,
                sort=args.sort,
                shuffle=args.shuffle,
            )

            test_train_iterator = BatchSampler(
                None,
                test_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                keep_last=True,
                dataset_is_array=True,
            )

            lrs_pbar.reset()
            lrs_pbar.n = 0
            lrs_pbar.refresh()

            best_lrs = []

            for lr in args.lr_sweep:
                rng = PRNGKey(traj_length + n + 1)
                # tmp_fast_params = get_test_fast_params(
                #     rng,
                #     state["fast_params"],
                #     zero_weight=args.zero_weight,
                #     keep_bias=args.keep_bias,
                # )

                test_train_loss, test_train_acc = test_sup_model.test(
                    test_train_iterator,
                    slow_params=tmp_slow_params,
                    fast_params=tmp_fast_params,
                    inner_lr=lr,
                    slow_state=state["state"],
                    fast_state=state["state"],
                )

                test_train_acc = test_train_acc.item()
                train_train_search_results_all[traj_length][lr].append(test_train_acc)

                # print(f"LR: {lr}, TRAJ LENGTH: {traj_length}, ACC: {test_train_acc}")

                lrs_pbar.update()
                lrs_pbar.refresh()

                best_lrs.append((lr, test_train_acc))

            best_lr = max(best_lrs, key=lambda x: x[1])
            train_train_search_results_bests[traj_length].append(best_lr)

            runs_pbar.update()
            runs_pbar.refresh()

        print("\n\n\n\nTRAJ LENGTH", traj_length, "RESULTS:")
        print(train_train_search_results_bests[traj_length])
        print()

    # BSZ = 256

    # # SORT = False
    # # SHUFFLE = True
    # N_RUNS_TEST = 50

    test_train_results = {t: [] for t in train_train_search_results_bests.keys()}
    test_test_results = {t: [] for t in train_train_search_results_bests.keys()}

    trajs_pbar = tqdm(train_train_search_results_bests.items())
    runs_pbar = tqdm(range(args.num_runs_test))

    for traj_length, traj_results in trajs_pbar:
        mode_lr = scipy.stats.mode([x[0] for x in traj_results]).mode.item()

        runs_pbar.reset()
        runs_pbar.n = 0
        runs_pbar.refresh()

        for n in range(args.num_runs_test):

            rng, random_state = split(PRNGKey(traj_length + n))

            test_train_dataset, test_test_dataset = sample_trajectory(
                random_state,
                test_dataset,
                test_dataset.targets,
                traj_length,
                sort=args.sort,
                shuffle=args.shuffle,
            )

            test_train_iterator = BatchSampler(
                None,
                test_train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                keep_last=True,
                dataset_is_array=True,
            )

            test_test_iterator = BatchSampler(
                None,
                test_test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                keep_last=True,
                dataset_is_array=True,
            )

            rng = PRNGKey(traj_length + n + 1)
            # tmp_fast_params = get_test_fast_params(
            #     rng,
            #     state["fast_params"],
            #     zero_weight=args.zero_weight,
            #     keep_bias=args.keep_bias,
            # )

            (test_train_loss, test_train_acc), (
                test_test_loss,
                test_test_acc,
            ) = test_sup_model.test(
                test_train_iterator,
                test_test_iterator,
                slow_params=tmp_slow_params,
                fast_params=tmp_fast_params,
                inner_lr=mode_lr,
                slow_state=state["state"],
                fast_state=state["state"],
            )

            test_train_acc = test_train_acc.item()
            test_test_acc = test_test_acc.item()

            test_train_results[traj_length].append((mode_lr, test_train_acc))
            test_test_results[traj_length].append((mode_lr, test_test_acc))

            runs_pbar.update()

        print(f"\n\n\n\nTRAJ LENGTH {traj_length} LEARNING RATE {mode_lr} RESULTS:")
        print(f"TRAIN ACC: {onp.mean([x[1] for x in test_train_results[traj_length]])}")
        print(f"TEST ACC: {onp.mean([x[1] for x in test_test_results[traj_length]])}")
        print()

    with open(
        os.path.join(
            args.model_dir,
            f"{args.model_filename}_results_shuffle-{args.shuffle}_sort-{args.sort}_zero-weight-{args.zero_weight}_keep-bias{args.keep_bias}.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "train": test_train_results,
                "test": test_test_results,
                "SHUFFLE": args.shuffle,
                "SORT": args.shuffle,
                "args": vars(args),
                "train_train_search_results_all": train_train_search_results_all,
                "train_train_search_results_bests": train_train_search_results_bests,
            },
            f,
            indent=4,
        )


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_filename", type=str, default="model.dill")
    parser.add_argument("--data_dir", type=str, default="../data/from_torch/omniglot")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_runs_search", type=int, default=10)
    parser.add_argument("--num_runs_test", type=int, default=50)
    parser.add_argument("--sort", required=True, type=int, choices=[0, 1])
    parser.add_argument("--shuffle", required=True, type=int, choices=[0, 1])
    parser.add_argument("--zero_weight", required=True, type=int, choices=[0, 1])
    parser.add_argument("--keep_bias", required=True, type=int, choices=[0, 1])

    parser.add_argument(
        "--lr_sweep",
        type=float,
        nargs="+",
        default=[
            #     0.01,
            # 0.0085,
            0.005,
            0.003,
            0.001,
            0.00085,
            0.0005,
            0.0003,
            0.0001,
                0.000085,
                0.00005,
            #     0.00003,
        ],
    )
    parser.add_argument(
        "--trajs",
        type=int,
        nargs="+",
        default=[
            10,
            50,
            75,
            100,
            200,
            300,
            400,
            600,
        ],
    )

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())