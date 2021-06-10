import os

import json

import dill
import numpy as onp
from tqdm import tqdm

from comet_ml import Experiment

import jax
from jax import numpy as jnp, random
from jax.random import split, PRNGKey
import haiku as hk
import optax as ox

import logging
from logging import handlers
from torch.utils.tensorboard import SummaryWriter

from dotenv import load_dotenv


from data import augment
from data.sampling import BatchSampler, MRCLDatasetLoader
from trainer.meta import CLWrapper
from models.resnet import resnet12
from losses import mean_xe_and_acc_dict


from datasets import get_dataset
from utils.data import Subset, ImageDataset

load_dotenv()

logger = logging.getLogger("experiment")


def make_mrcl_loader(
    rng,
    image_dataset,
    num_images_per_class,
    image_size,
    *args,
    **kwargs,
):
    # image_size = image_dataset.images.shape[-2]
    return MRCLDatasetLoader(
        rng,
        image_dataset.inputs.reshape(
            -1, num_images_per_class, image_size, image_size, 1
        ),
        image_dataset.targets.reshape(-1, num_images_per_class),
        *args,
        **kwargs,
    )


def sample_trajectory(
    rng,
    dataset,
    targets,
    traj_length,
    num_train_samples=15,
    sort=False,
    shuffle=True,
):
    rng, rng_classes = split(rng)
    selected_classes = random.choice(
        rng_classes, onp.unique(targets), (traj_length,), replace=False
    )

    if sort:
        selected_classes = sorted(selected_classes)

    train_indexes = []
    test_indexes = []
    for _class in selected_classes:
        _indexes = onp.nonzero(_class == targets)[0]
        if shuffle:
            rng, rng_shuffle = split(rng)

            _indexes = random.permutation(rng_shuffle, _indexes)

        train_indexes.append(_indexes[:num_train_samples])
        test_indexes.append(_indexes[num_train_samples:])

    train_dataset = Subset(
        dataset,
        onp.concatenate(train_indexes),
    )
    test_dataset = Subset(dataset, onp.concatenate(test_indexes))

    return train_dataset, test_dataset


def make_test_iterators(
    rng,
    test_dataset,
    targets,
    n,
    classes,
    traj_length,
    sort,
    shuffle,
):
    test_train_iterators, test_test_iterators = [], []
    for _ in range(n):
        rng, rng_classes = split(rng)

        test_train_dataset, test_test_dataset = sample_trajectory(
            rng_classes,
            test_dataset,
            targets,
            traj_length,
            sort=sort,
            shuffle=shuffle,
        )

        test_train_iterator = BatchSampler(
            random.PRNGKey(0),  # Since we do not shuffle the random key does not matter
            test_train_dataset,
            128,
            shuffle=False,
            keep_last=True,
            dataset_is_array=True,
        )

        test_test_iterator = BatchSampler(
            random.PRNGKey(0),
            test_test_dataset,
            128,
            shuffle=False,
            keep_last=True,
            dataset_is_array=True,
        )

        test_train_iterators.append(test_train_iterator)
        test_test_iterators.append(test_test_iterator)

    return test_train_iterators, test_test_iterators


kaiming_normal = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")


class OMLConvnet(hk.Module):
    def __init__(
        self,
        output_size,
        spatial_dims,
        image_size=84,
        conv_hidden_size=256,
        normalize=None,
        activation=None,
        name=None,
    ):
        super().__init__(name=name)
        self.image_size = image_size
        self.hidden_size = conv_hidden_size
        self.spatial_dims = spatial_dims
        if image_size == 84:
            self.strides = (2, 1, 2, 1, 2, 2)
        elif image_size == 28:
            self.strides = (1, 1, 2, 1, 1, 2)

    def __call__(self, x, mask=None, phase="all", training=None):
        assert phase in ["all", "encoder", "adaptation"]

        if (phase == "all") or (phase == "encoder"):
            for stride in self.strides:
                x = hk.Conv2D(
                    output_channels=self.hidden_size,
                    kernel_shape=3,
                    stride=stride,
                    padding="VALID",
                    w_init=kaiming_normal,
                )(x)
                x = jax.nn.relu(x)
            x = hk.Reshape((self.hidden_size * self.spatial_dims,))(x)
        if (phase == "all") or (phase == "adaptation"):
            x = hk.Linear(
                1000,
                w_init=kaiming_normal,
            )(x)
        else:
            x = (x,)

        return x


class ANMLNet(hk.Module):
    def __init__(
        self,
        output_size,
        spatial_dims,
        image_size=None,  # Compatibility
        normalize=None,
        activation=None,
        name=None,
    ):
        super().__init__(name=name)
        self.output_size = output_size
        self.spatial_dims = spatial_dims

    def __call__(self, inp, mask=None, phase="all", training=None):
        assert phase in [
            "all",
            "encoder",
            "adaptation",
            "test_encoder",
            "test_adaptation",
            "everything"
        ]

        # NM branch
        if phase in ["all", "encoder", "test_encoder", "everything"]:
            with hk.experimental.name_scope("nm"):
                x = inp
                for i in range(3):
                    x = hk.Conv2D(
                        output_channels=112,
                        kernel_shape=3,
                        stride=1,
                        padding="VALID",
                        w_init=kaiming_normal,
                    )(x)
                    x = hk.InstanceNorm(
                        True,
                        True,
                    )(x)
                    x = jax.nn.relu(x)
                    if i != 2:
                        x = hk.max_pool(
                            x,
                            window_shape=(1, 2, 2, 1),
                            strides=(1, 2, 2, 1),
                            padding="VALID",
                        )
                x = hk.Reshape((112 * self.spatial_dims,))(x)
                x = hk.Linear(
                    2304,
                    w_init=kaiming_normal,
                )(x)
                mask = jax.nn.sigmoid(x)

        if phase in [
            "all",
            "adaptation",
            "test_adaptation",
            "test_encoder",
            "everything",
        ]:

            if phase in [
                "all",
                "adaptation",
                "test_encoder",
                "everything",
            ]:
                with hk.experimental.name_scope("rln"):

                    x = inp
                    for i in range(3):
                        x = hk.Conv2D(
                            output_channels=256,
                            kernel_shape=3,
                            stride=1,
                            padding="VALID",
                            w_init=kaiming_normal,
                        )(x)
                        x = hk.InstanceNorm(
                            True,
                            True,
                        )(x)
                        x = jax.nn.relu(x)
                        if i != 2:
                            x = hk.max_pool(
                                x,
                                window_shape=(1, 2, 2, 1),
                                strides=(1, 2, 2, 1),
                                padding="VALID",
                            )
                    x = hk.Reshape((256 * self.spatial_dims,))(x)
                    pre_nm = x
                    x = x * mask
                    post_nm = x

            elif phase in ["test_adaptation"]:
                x = inp

            if phase in [
                "all",
                "adaptation",
                "test_adaptation",
                "everything",
            ]:
                with hk.experimental.name_scope("classifier"):
                    x = hk.Linear(
                        self.output_size,
                        w_init=kaiming_normal,
                    )(x)
                    pred = x
            elif phase in ["test_encoder"]:
                x = (x,)
        else:
            x = (inp, mask)

        if phase == "everything":
            return (mask, pre_nm, post_nm, pred)

        return x


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
        args.dataset,
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
        # if args.model == "oml":
        #     num_tasks = 963
        # else:
        #     num_tasks = 964

    test_dataset = get_dataset(
        args.dataset,
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

    if "resnet" in args.model:
        _model = resnet12
    elif args.model == "anml":
        _model = ANMLNet
    elif args.model == "oml":
        _model = OMLConvnet

    # if "anml" in args.model:
    if args.train_method == "anml":
        inner_subset = train_dataset
        outer_subset = train_dataset
        mrcl_loader_kwargs = dict(
            way=1,
            shot=20,
            qry_shot=0,
            cl_qry_shot=0,
            batch_size=bsz,
            disjoint=True,
        )

    # elif "oml" in args.model:
    elif args.train_method == "oml":
        inner_subset = train_dataset.get_classes_subset(
            list(range(int(num_tasks / 2), num_tasks))
        )
        outer_subset = train_dataset.get_classes_subset(
            list(range(0, int(num_tasks / 2)))
        )

        mrcl_loader_kwargs = dict(
            way=3,
            shot=5,
            qry_shot=5,
            cl_qry_shot=0,
            batch_size=bsz,
            disjoint=True,
        )

    else:
        _model = None

    logger.info(f"Inner train dataset size: {len(inner_subset)}")
    logger.info(f"Outer train dataset size: {len(outer_subset)}")

    mrcl_loader = make_mrcl_loader(
        rng_data1,
        inner_subset,
        20,
        args.image_size,
        **mrcl_loader_kwargs,
    )
    # if "oml" in args.model:
    if args.train_method == "oml":
        outer_bsz = 15 * bsz
    # elif "anml" in args.model:
    elif args.train_method == "anml":
        outer_bsz = 64 * bsz
    complete_iterator = BatchSampler(
        rng_data2,
        outer_subset,
        outer_bsz,
        shuffle=True,
        keep_last=False,
        dataset_is_array=True,
    )

    model = hk.transform_with_state(
        lambda x, mask=None, phase="all", training=None: _model(
            1000,
            9,
            image_size=args.image_size,
            normalize=args.normalize,
            activation=args.activation,
        )(
            x,
            mask=mask,
            phase=phase,
            training=training,
        )
    )

    if args.normalize_input:
        normalize_fn = train_dataset.normalize
    else:
        normalize_fn = lambda x: x

    dummy_input, dummy_targets = next(
        iter(complete_iterator)
    )  # only uses shape to initialize parameters
    print(dummy_input.shape)
    slow_params, slow_state = model.init(
        rng_slow, normalize_fn(dummy_input / 255), phase="encoder"
    )
    dummy_output, _ = model.apply(
        slow_params,
        slow_state,
        split(rng_slow)[0],
        normalize_fn(dummy_input / 255),
        phase="encoder",
        training=True,
    )
    fast_params, fast_state = model.init(
        rng_fast,
        *dummy_output,
        phase="adaptation",
    )

    ## Test data

    # lowest_target = test_train_dataset.targets.min()
    # test_train_dataset.targets = test_train_dataset.targets - lowest_target
    # test_test_dataset.targets = test_test_dataset.targets - lowest_target

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

    optimizer = ox.scale_by_adam()

    def inner_opt_update_fn(lr, updates, state, params):
        inner_opt = ox.sgd(lr)
        return inner_opt.update(updates, state, params)

    # def get_step_lr(step_num, lr, sch_dict):
    #     # return ox.cosine_decay_schedule(lr, args.steps, args.cosine_mult)(step_num)
    #     return jnp.asarray(lr)
    def get_step_lr(step_num, lr, sch_dict):
        return ox.piecewise_constant_schedule(lr, sch_dict)(step_num)

    def schedule(step_num, updates, lr, sch_dict=None):
        return ox.scale(get_step_lr(step_num, lr, sch_dict)).update(
            updates,
            None,
        )[0]

    test_optimizer = ox.adam(0)

    def test_inner_opt_update_fn(lr, updates, state, params):
        inner_opt = ox.adam(lr)
        return inner_opt.update(updates, state, params)

    def reset_fast_params_fn(w_make_fn, rng, spt_classes, params, all=False):
        if args.model == "anml":
            w_name = "anml_net/classifier/linear"
        elif args.model == "oml":
            w_name = "oml_convnet/linear"
        elif "resnet" in args.model:
            w_name = "res_net/classifier"
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

    inner_lr = args.inner_lr
    lr = jnp.array(args.outer_lr)
    sch_dict = args.schedule

    # normalize_fn = train_dataset.normalize
    # normalize_fn = lambda x: x

    if args.train_reset == "random":
        reset_fn = jax.nn.initializers.he_normal
    elif args.train_reset == "zero":
        reset_fn = lambda dtype: lambda rng, shape: jax.nn.initializers.zeros(
            rng,
            shape,
            dtype=dtype,
        )
    # qry_preprocess_fn = spt_preprocess_fn = lambda rng, inputs: normalize_fn(
    #     inputs / 255
    # )
    # test_preprocess_fn = lambda inputs: normalize_fn(inputs / 255)
    qry_preprocess_fn = spt_preprocess_fn = lambda rng, inputs: normalize_fn(
        inputs / 255
    )
    test_preprocess_fn = lambda inputs: normalize_fn(inputs / 255)
    if args.augment == "none":
        logger.info("No augmentation")
    elif args.augment == "qry":
        logger.info("Augment qry")
        qry_preprocess_fn = lambda rng, inputs: normalize_fn(
            augment(rng, inputs / 255, out_size=args.image_size)
        )
    elif args.augment == "spt":
        logger.info("Augment spt")
        spt_preprocess_fn = lambda rng, inputs: normalize_fn(
            augment(rng, inputs / 255, out_size=args.image_size)
        )
    elif args.augment == "all":
        logger.info("Augment all")
        qry_preprocess_fn = spt_preprocess_fn = lambda rng, inputs: normalize_fn(
            augment(rng, inputs / 255, out_size=args.image_size)
        )

    if args.model == "anml":
        val_slow_apply = jax.partial(model.apply, phase="test_encoder")
        val_fast_apply = jax.partial(model.apply, phase="test_adaptation")
    else:
        val_slow_apply = None
        val_fast_apply = None

    trainer = CLWrapper(
        slow_apply=jax.partial(model.apply, phase="encoder"),
        fast_apply=jax.partial(model.apply, phase="adaptation"),
        val_slow_apply=val_slow_apply,
        val_fast_apply=val_fast_apply,
        slow_params=slow_params,
        fast_params=fast_params,
        slow_state=slow_state,
        fast_state=fast_state,
        inner_lr=inner_lr,
        loss_fn=mean_xe_and_acc_dict,
        optimizer=optimizer,
        scheduler=jax.partial(schedule, lr=-lr, sch_dict=sch_dict),
        init_inner_opt_state_fn=ox.sgd(0).init,
        inner_opt_update_fn=inner_opt_update_fn,
        reset_fast_params_fn=jax.partial(reset_fast_params_fn, reset_fn),
        preprocess_qry_fn=qry_preprocess_fn,
        preprocess_spt_fn=spt_preprocess_fn,
        preprocess_test_fn=test_preprocess_fn,
        test_init_inner_opt_state_fn=test_optimizer.init,
        test_inner_opt_update_fn=test_inner_opt_update_fn,
        reset_before_outer_loop=bool(args.reset_before_outer_loop),
    )

    trainer.init_opt_state()
    step_fn = jax.jit(
        lambda step_num, x_spt, y_spt, x_qry, y_qry, opt_state, rng, spt_classes, slow_params, fast_params, slow_state, fast_state, inner_lr: trainer.single_step(
            step_num,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            opt_state=opt_state,
            rng=rng,
            spt_classes=spt_classes,
            slow_params=slow_params,
            fast_params=fast_params,
            slow_state=slow_state,
            fast_state=fast_state,
            inner_lr=inner_lr,
        )
    )
    trainer.set_step_fn(step_fn)

    if (args.model == "anml") or ("resnet" in args.model):

        def get_test_slow_params(trainer):
            return hk.data_structures.merge(
                trainer.slow_params,
                hk.data_structures.partition(
                    lambda module_name, name, value: "classifier" in module_name,
                    trainer.fast_params,
                )[1],
            )

        def get_test_fast_params(
            rng,
            trainer,
            zero_weight=args.test_zero_weight,
            keep_bias=args.test_keep_bias,
        ):
            tmp_fast_params = hk.data_structures.partition(
                lambda module_name, name, value: "classifier" in module_name,
                trainer.fast_params,
            )[0]

            weights, biases = hk.data_structures.partition(
                lambda module_name, name, value: name == "w", tmp_fast_params
            )
            tmp_fast_params = hk.data_structures.merge(
                jax.tree_map(lambda t: jnp.zeros_like(t), weights)
                if zero_weight
                else jax.tree_map(
                    lambda t: reset_fn(dtype=t.dtype)(rng, t.shape), weights
                ),
                biases
                if keep_bias
                else jax.tree_map(lambda t: jnp.zeros_like(t), biases),
            )
            return tmp_fast_params

    elif args.model == "oml":

        def get_test_slow_params(trainer):
            return trainer.slow_params

        def get_test_fast_params(
            rng,
            trainer,
            zero_weight=args.test_zero_weight,
            keep_bias=args.test_keep_bias,
        ):
            tmp_fast_params = trainer.fast_params
            weights, biases = hk.data_structures.partition(
                lambda module_name, name, value: name == "w", tmp_fast_params
            )
            tmp_fast_params = hk.data_structures.merge(
                jax.tree_map(lambda t: jnp.zeros_like(t), weights)
                if zero_weight
                else jax.tree_map(
                    lambda t: reset_fn(dtype=t.dtype)(rng, t.shape), weights
                ),
                biases
                if keep_bias
                else jax.tree_map(lambda t: jnp.zeros_like(t), biases),
            )
            return tmp_fast_params

    trainer.test(
        test_train_iterators[0],
        test_test_iterators[0],
        inner_lr=0.00085,
        slow_params=get_test_slow_params(trainer),
        fast_params=get_test_fast_params(PRNGKey(0), trainer),
    )

    # lr_sweep = [
    #     #     0.03,
    #     0.01,
    #     0.003,
    #     0.001,
    #     0.00085,
    #     #     0.0006,
    #     # 0.0004,
    #     0.0003,
    #     0.0001,
    #     0.000085,
    #     # 0.00006,
    #     0.00003,
    #     # 0.00001,
    # ]
    lr_sweep = args.lr_sweep

    histories = {
        "train": {
            "iil": [],
            "fil": [],
            "iia": [],
            "fia": [],
            "iol": [],
            "fol": [],
            "ioa": [],
            "foa": [],
            "lr": [],
            "steps": [],
        },
        "train_special": {
            "inner_loss_progress": [],
            "inner_acc_progress": [],
        },
        "val": {
            "loss_train": {lr: [] for lr in lr_sweep},
            "acc_train": {lr: [] for lr in lr_sweep},
            "loss_test": {lr: [] for lr in lr_sweep},
            "acc_test": {lr: [] for lr in lr_sweep},
            "step": [],
        },
    }

    best_val_acc = 0
    loss_ema = aux_ema = None
    # iterator = iter(mrcl_loader)
    n_steps = args.num_steps
    train_pbar = tqdm(
        range(1, n_steps + 1),
        ncols=0,
        mininterval=2.5,
    )

    writer = SummaryWriter(os.path.join(logdir, "tensorboard"))

    for step_num in train_pbar:
        x_spt, y_spt, _x_qry, _y_qry, _, _ = next(mrcl_loader)

        x_qry_cl, y_qry_cl = next(iter(complete_iterator))

        # if "oml" in args.model:
        if args.train_method == "oml":
            x_qry = onp.concatenate(
                (_x_qry, x_qry_cl.reshape(bsz, -1, *x_qry_cl.shape[1:])), axis=1
            )
            y_qry = onp.concatenate((_y_qry, y_qry_cl.reshape(bsz, -1)), axis=1)
        # elif "anml" in args.model:
        elif args.train_method == "anml":
            x_qry = onp.concatenate(
                (x_spt, x_qry_cl.reshape(bsz, -1, *x_qry_cl.shape[1:])), axis=1
            )
            y_qry = onp.concatenate((y_spt, y_qry_cl.reshape(bsz, -1)), axis=1)

        rng, rng_step = split(rng)
        spt_classes = onp.unique(y_spt, axis=-1)
        loss, aux = trainer.train_step(
            step_num - 1,
            rng_step,
            x_spt,
            y_spt,
            x_qry,
            y_qry,
            spt_classes,
        )

        aux = jax.tree_map(jax.partial(jnp.mean, axis=0), aux)
        if loss_ema is None:
            loss_ema = loss
            aux_ema = aux
        else:
            (loss_ema, aux_ema) = jax.tree_multimap(
                lambda ema, x: ema * 0.9 + x * 0.1,
                (loss_ema, aux_ema),
                (loss, aux),
            )

        if (
            (((step_num) % 100) == 0)
            or ((step_num) == n_steps)
            or (step_num == 1)
            or ((step_num % args.test_interval) == 0)
        ):

            loss = fol = loss_ema.item()

            iil = aux_ema["inner"][0]["initial"]["loss"].item()
            fil = aux_ema["inner"][0]["final"]["loss"].item()
            iia = aux_ema["inner"][0]["initial"]["aux"]["acc"].item()
            fia = aux_ema["inner"][0]["final"]["aux"]["acc"].item()

            iol = aux_ema["outer"]["initial"]["loss"].item()
            fol = aux_ema["outer"]["final"]["loss"].item()
            ioa = aux_ema["outer"]["initial"]["acc"].item()
            foa = aux_ema["outer"]["final"]["acc"].item()

            inner_loss_progress = aux_ema["inner"][1].tolist()
            inner_acc_progress = aux_ema["inner"][2]["acc"].tolist()

            train_pbar.set_postfix(
                loss=loss,
                iil=iil,
                fil=fil,
                iia=iia,
                fia=fia,
                iol=iol,
                ioa=ioa,
                foa=foa,
            )

            histories["train"]["iil"].append(iil)
            histories["train"]["fil"].append(fil)
            histories["train"]["iia"].append(iia)
            histories["train"]["fia"].append(fia)
            histories["train"]["iol"].append(iol)
            histories["train"]["fol"].append(fol)
            histories["train"]["ioa"].append(ioa)
            histories["train"]["foa"].append(foa)

            histories["train_special"]["inner_loss_progress"].append(
                inner_loss_progress
            )
            histories["train_special"]["inner_acc_progress"].append(inner_acc_progress)

            histories["train"]["steps"].append(step_num)
            step_lr = get_step_lr(
                step_num - 1,
                lr=lr,
                sch_dict=sch_dict,
            ).item()
            histories["train"]["lr"].append(step_lr)

            train_metrics = {
                "lr": step_lr,
                "loss": loss,
                "iil": iil,
                "fil": fil,
                "iia": iia,
                "fia": fia,
                "iol": iol,
                "ioa": ioa,
                "foa": foa,
            }

            for tag, metric in train_metrics.items():
                writer.add_scalar(
                    "train/" + tag,
                    metric,
                    global_step=step_num,
                )

            # writer.add_scalar(, , global_step=step_num + 1)
            # writer.add_scalar("train/loss", loss, global_step=step_num + 1)
            # writer.add_scalar("train/iil", iil, global_step=step_num + 1)
            # writer.add_scalar("train/fil", fil, global_step=step_num + 1)
            # writer.add_scalar("train/iia", iia, global_step=step_num + 1)
            # writer.add_scalar("train/fia", fia, global_step=step_num + 1)
            # writer.add_scalar("train/iol", iol, global_step=step_num + 1)
            # writer.add_scalar("train/ioa", ioa, global_step=step_num + 1)
            # writer.add_scalar("train/foa", foa, global_step=step_num + 1)

            comet.log_metrics(
                train_metrics,
                prefix="train",
                step=step_num,
            )

            # experiment.log_metrics(
            #    {k: v[-1] for k, v in histories["train"].items()}, step=step_num
            # )

            # print(histories)

        if (
            ((step_num) % args.test_interval) == 0
            or ((step_num) == n_steps)
            or (step_num == 1)
        ):
            histories["val"]["step"].append(step_num)
            logger.info("\nTrain metrics")
            logger.info(str(train_metrics))
            logger.info(f"TEST STEP {step_num}")

            train_accs = {}
            test_accs = {}
            train_losses = {}
            test_losses = {}
            tmp_slow_params = get_test_slow_params(trainer)

            for inner_lr in lr_sweep:
                train_accs[inner_lr] = []
                test_accs[inner_lr] = []
                train_losses[inner_lr] = []
                test_losses[inner_lr] = []

                rng, rng_its = split(rng)
                # test_train_iterators, test_test_iterators = make_test_iterators(
                #     rng_its,
                #     test_train_dataset,
                #     test_test_dataset,
                #     5,
                #     test_classes_arange,
                #     test_traj_length,
                # )

                for test_train_iterator, test_test_iterator in zip(
                    test_train_iterators, test_test_iterators
                ):
                    rng, rng_test = split(rng)
                    tmp_fast_params = get_test_fast_params(
                        rng_test,
                        trainer,
                    )
                    (
                        (test_train_loss, test_train_acc),
                        (test_test_loss, test_test_acc),
                    ) = trainer.test(
                        test_train_iterator,
                        test_test_iterator,
                        inner_lr=inner_lr,
                        slow_params=tmp_slow_params,
                        fast_params=tmp_fast_params,
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
                global_step=step_num,
            )
            writer.add_scalars(
                "test/loss_test",
                {
                    str(lr): onp.mean(v[-1])
                    for lr, v in histories["val"]["loss_test"].items()
                },
                global_step=step_num,
            )
            writer.add_scalars(
                "test/acc_train",
                {
                    str(lr): onp.mean(v[-1])
                    for lr, v in histories["val"]["acc_train"].items()
                },
                global_step=step_num,
            )
            writer.add_scalars(
                "test/acc_test",
                {
                    str(lr): onp.mean(v[-1])
                    for lr, v in histories["val"]["acc_test"].items()
                },
                global_step=step_num,
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
                step=step_num,
            )

            # logger.info(str(step_num))
            # logger.info(str([t.item() for t in train_accs]))
            # logger.info(str([t.item() for t in train_losses]))
            # logger.info(str([t.item() for t in test_accs]))
            # logger.info(str([t.item() for t in test_losses]))

            # test_train_loss = onp.mean([t.item() for t in train_losses])
            # test_test_loss = onp.mean([t.item() for t in test_losses])
            # test_train_acc = onp.mean([t.item() for t in train_accs])
            # test_test_acc = onp.mean([t.item() for t in test_accs])

            # logger.info(
            #     f"mean train loss: {test_train_loss}, mean train acc: {test_train_acc}"
            # )
            # logger.info(
            #     f"mean test loss: {test_test_loss} mean test acc: {test_test_acc}"
            # )

            # writer.add_scalars(
            #     "test/train_loss", test_train_loss, global_step=step_num + 1
            # )
            # writer.add_scalar(
            #     "test/train_acc", test_train_acc, global_step=step_num + 1
            # )
            # writer.add_scalar(
            #     "test/test_loss", test_test_loss, global_step=step_num + 1
            # )
            # writer.add_scalar("test/test_acc", test_test_acc, global_step=step_num + 1)

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
                            "slow_params": trainer.slow_params,
                            "fast_params": trainer.fast_params,
                            "slow_state": trainer.slow_state,
                            "fast_state": trainer.fast_state,
                            "opt_state": trainer.opt_state,
                            "inner_lr": best_lr,
                            # "test_train_acc": test_train_acc,
                            # "test_test_acc": test_test_acc,
                            # "test_train_loss": test_train_loss,
                            # "test_test_loss": test_test_loss,
                        },
                        f,
                    )

        if ((step_num) % 5000) == 0:
            with open(os.path.join(logdir, "histories.json"), "w") as f:
                json.dump(histories, f)

            with open(os.path.join(logdir, "model.dill"), "wb") as f:
                dill.dump(
                    {
                        "histories": histories,
                        "slow_params": trainer.slow_params,
                        "fast_params": trainer.fast_params,
                        "slow_state": trainer.slow_state,
                        "fast_state": trainer.fast_state,
                        "opt_state": trainer.opt_state,
                        "histories": histories,
                        "best_val_acc": best_val_acc,
                    },
                    f,
                )
        if ((step_num) == 20000) and (args.model == "anml"):
            with open(os.path.join(logdir, "histories_20k.json"), "w") as f:
                json.dump(histories, f)
            with open(os.path.join(logdir, "model_20k.dill"), "wb") as f:
                dill.dump(
                    {
                        "histories": histories,
                        "slow_params": trainer.slow_params,
                        "fast_params": trainer.fast_params,
                        "slow_state": trainer.slow_state,
                        "fast_state": trainer.fast_state,
                        "opt_state": trainer.opt_state,
                        "histories": histories,
                        "best_val_acc": best_val_acc,
                        # "last_val_acc":
                        # "test_train_acc": test_train_acc,
                        # "test_test_acc": test_test_acc,
                    },
                    f,
                )
        if ((step_num) % 200000) == 0:
            with open(os.path.join(logdir, f"histories_{step_num}.json"), "w") as f:
                json.dump(histories, f)
            with open(os.path.join(logdir, f"model_{step_num}.dill"), "wb") as f:
                dill.dump(
                    {
                        "histories": histories,
                        "slow_params": trainer.slow_params,
                        "fast_params": trainer.fast_params,
                        "slow_state": trainer.slow_state,
                        "fast_state": trainer.fast_state,
                        "opt_state": trainer.opt_state,
                        "histories": histories,
                        "best_val_acc": best_val_acc,
                        # "last_val_acc":
                        # "test_train_acc": test_train_acc,
                        # "test_test_acc": test_test_acc,
                    },
                    f,
                )
        if (step_num) == 700000:
            with open(os.path.join(logdir, "histories_700k.json"), "w") as f:
                json.dump(histories, f)
            with open(os.path.join(logdir, "model_700k.dill"), "wb") as f:
                dill.dump(
                    {
                        "histories": histories,
                        "slow_params": trainer.slow_params,
                        "fast_params": trainer.fast_params,
                        "slow_state": trainer.slow_state,
                        "fast_state": trainer.fast_state,
                        "opt_state": trainer.opt_state,
                        # "test_train_acc": test_train_acc,
                        # "test_test_acc": test_test_acc,
                    },
                    f,
                )

    with open(os.path.join(logdir, "histories.json"), "w") as f:
        json.dump(histories, f)

    with open(os.path.join(logdir, "model.dill"), "wb") as f:
        dill.dump(
            {
                "histories": histories,
                "slow_params": trainer.slow_params,
                "fast_params": trainer.fast_params,
                "slow_state": trainer.slow_state,
                "fast_state": trainer.fast_state,
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
    parser.add_argument(
        "--dataset", default="omniglot", choices=["omniglot", "tiered-imagenet"]
    )
    parser.add_argument("--schedule", nargs="*", action=keyvalue, default={})
    parser.add_argument("--data_dir", type=str, default="../data/from_torch/omniglot")

    parser.add_argument("--normalize", choices=["bn", "in"])
    parser.add_argument("--activation", choices=["relu", "leaky_relu"], default="relu")
    parser.add_argument("--normalize_input", default=False, action="store_true")
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_interval", type=int, required=True)
    parser.add_argument("--sorted_test", type=int, default=0, choices=[0, 1])
    parser.add_argument("--shuffle_test", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--augment", default="none", choices=["none", "all", "spt", "qry"]
    )

    parser.add_argument(
        "--train_reset", type=str, required=True, choices=["random", "zero"]
    )
    parser.add_argument(
        "--reset_before_outer_loop", type=int, default=1, choices=[0, 1]
    )

    parser.add_argument("--inner_lr", type=float, required=True)
    parser.add_argument("--outer_lr", type=float, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=["oml", "anml", "resnet-oml", "resnet-anml"],
        required=True,
    )
    parser.add_argument(
        "--train_method",
        type=str,
        choices=["oml", "anml"],
        required=True,
    )
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
            # 0.01,
            0.0085,
            0.005,
            0.003,
            0.001,
            0.00085,
            0.0005,
            0.0003,
            0.0001,
            # 0.000085,
            # 0.00005,
            # 0.00003,
        ],
    )

    args = parser.parse_args()
    args.schedule = {int(k): float(v) for k, v in args.schedule.items()}
    return args


if __name__ == "__main__":
    main(parse_args())
