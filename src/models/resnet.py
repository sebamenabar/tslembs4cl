import jax
import haiku as hk

kaiming_normal = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")


def conv3x3(
    # in_planes,
    out_planes,
    initializer,
    stride=1,
):
    return hk.Conv2D(
        output_channels=out_planes,
        kernel_shape=3,
        stride=stride,
        padding="SAME",
        with_bias=False,
        w_init=initializer,
    )


def get_norm(norm_name, **kwargs):
    if norm_name == "bn":
        # print("Using batch normalization")
        norm = lambda name: hk.BatchNorm(**kwargs, name=name)
    elif norm_name == "in":
        norm = lambda name: hk.InstanceNorm(True, True, name=name)
    # elif norm_name == "custom":
    #     # print("Using my batch normalization")
    #     norm = lambda name: MyBatchNorm(**kwargs, name=name)
    # elif norm_name == "affine":
    #     # print("Using affine normalization")
    #     norm = lambda name: Affine(name=name)
    # elif norm_name == "none":
    #     norm = lambda name: lambda x, _: x

    return norm

class BasicBlock(hk.Module):
    expansion = 1
    # Based on https://github.com/WangYueFt/rfs/blob/master/models/resnet.py
    # and https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/nets/resnet.py
    def __init__(
        self,
        # inplanes,
        planes,  # channels
        bn_config,
        use_projection,
        stride=1,
        # drop_rate=0.0,
        # drop_block=False,
        # block_size=1,
        use_se=False,
        w_initializer="glorot_uniform",
        activation="leaky_relu",
        normalize="bn",
        learn_residual_step=False,
        name=None,
        # norm_init_zero=False,
    ):
        super().__init__(name=name)
        if activation == "relu":
            self.activation = jax.nn.relu
        elif activation == "leaky_relu":
            self.activation = jax.nn.leaky_relu
        self.learn_residual_step = learn_residual_step

        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault(
            "decay_rate",
            0.9,  # Pytorch uses 0.9 by default, Haiku ResNet had 0.999
        )

        # print("batch norm config", bn_config)
        norm = get_norm(normalize, **bn_config)

        self.normalize = normalize
        # if normalize == "bn":
        #     # print("Using batch normalization")
        #     norm = lambda name: hk.BatchNorm(**bn_config, name=name)
        # elif normalize == "custom":
        #     # print("Using my batch normalization")
        #     norm = lambda name: MyBatchNorm(**bn_config, name=name)
        # elif normalize == "affine":
        #     # print("Using affine normalization")
        #     norm = lambda name: Affine(name=name)

        # w_init = build_initializer(
        #     nonlinearity=activation, name=w_initializer, truncated=False
        # )
        w_init = kaiming_normal

        self.relu = self.activation  #
        self.conv1 = conv3x3(planes, w_init)
        self.bn1 = norm(None)
        self.conv2 = conv3x3(planes, w_init)
        self.bn2 = norm(None)
        self.conv3 = conv3x3(planes, w_init)
        self.bn3 = norm(None)
        self.maxpool = hk.MaxPool(
            window_shape=stride,
            strides=stride,
            padding="VALID",
            channel_axis=-1,
        )
        self.use_projection = use_projection
        if self.use_projection:
            self.downsample_conv = hk.Conv2D(
                output_channels=planes,
                kernel_shape=1,
                stride=1,
                padding="VALID",
                with_bias=False,
                w_init=w_init,
                name="shortcut_conv",
            )
            bn_config["scale_init"] = None
            norm = get_norm(normalize, **bn_config)
            self.downsample_bn = norm("shortcut_bn")

        self.stride = stride
        # self.drop_rate = None
        # self.num_batches_tracked = 0
        # self.drop_block = drop_block
        # self.block_size = block_size
        # self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        # if self.use_se:
        #    self.se = SELayer(plaanes 4)

    def __call__(self, x, is_training, test_local_stats):
        residual = x

        out = self.conv1(x)
        if (self.normalize == "bn") or (self.normalize == "custom"):
            out = self.bn1(out, is_training, test_local_stats)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if (self.normalize == "bn") or (self.normalize == "custom"):
            out = self.bn2(out, is_training, test_local_stats)
        else:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if (self.normalize == "bn") or (self.normalize == "custom"):
            out = self.bn3(out, is_training, test_local_stats)
        else:
            out = self.bn3(out)

        # if self.use_se:
        #     out = self.se(out)

        if self.use_projection:
            residual = self.downsample_conv(residual)
            if (self.normalize == "bn") or (self.normalize == "custom"):
                residual = self.downsample_bn(residual, is_training, test_local_stats)
            else:
                residual = self.downsample_bn(residual)
            # residual = self.downsample_bn(residual, is_training, test_local_stats)
        
        if self.learn_residual_step:
            # print("Learning step size")
            step_size = hk.get_parameter("residual_step", [], out.dtype, hk.initializers.Constant(0.))
            out = residual + step_size * out
        else:
            out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        return out


class BlockGroup(hk.Module):
    def __init__(
        self,
        block_cls,
        planes,
        num_blocks,
        stride,
        bn_config,
        w_initializer,
        activation,
        use_projection,
        normalize="bn",
        learn_residual_step=False,
        name=None,
    ):
        super().__init__(name=name)


        self.blocks = []
        if num_blocks == 1:
            pass
        else:
            pass
        self.blocks.append(
            block_cls(
                planes=planes,
                stride=stride,
                use_projection=use_projection,
                bn_config=bn_config,
                w_initializer=w_initializer,
                activation=activation,
                normalize=normalize,
                learn_residual_step=learn_residual_step,
                # drop_rate, drop_block, block_size, use_se
            )
        )

        for i in range(1, num_blocks):
            if i == num_blocks - 1:
                layer = block_cls(
                    planes=planes,
                    use_projection=False,
                    bn_config=bn_config,
                    w_initializer=w_initializer,
                    activation=activation,
                    normalize=normalize,
                    learn_residual_step=learn_residual_step,
                    # drop_rate=drop_rate,
                    # drop_block=drop_block,
                    # block_size=block_size,
                    # use_se=self.use_se,
                )
            else:
                layer = block_cls(
                    planes=planes,
                    use_projection=False,
                    bn_config=bn_config,
                    w_initializer=w_initializer,
                    activation=activation,
                    normalize=normalize,
                    learn_residual_step=learn_residual_step,
                )

            self.blocks.append(layer)

    def __call__(self, inputs, is_training, test_local_stats):
        out = inputs
        for i, block in enumerate(self.blocks):
            out = block(out, is_training, test_local_stats)
        return out


class ResNet(hk.Module):
    def __init__(
        self,
        output_size,
        spatial_dim, # compatibility
        block_cls,
        num_blocks,
        image_size=None, # compatibility
        bn_config=None,
        keep_prob=1.0,
        avg_pool=False,
        drop_rate=0.0,
        dropblock_size=5,
        num_classes=-1,
        use_se=False,
        w_initializer="glorot_uniform",
        activation="relu",
        normalize="bn",
        learn_residual_step=False,
        name=None,
    ):
        super().__init__(name=name)
        self.inplanes = 3
        self.use_se = use_se
        self.avg_pool = avg_pool
        self.output_size = output_size

        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        self.layer1 = self._make_layer(
            block_cls,
            num_blocks[0],
            64,
            bn_config,
            w_initializer,
            activation,
            stride=2,
            name="layer1",
            normalize=normalize,
            learn_residual_step=learn_residual_step,
        )
        self.layer2 = self._make_layer(
            block_cls,
            num_blocks[1],
            160,
            bn_config,
            w_initializer,
            activation,
            stride=2,
            name="layer2",
            normalize=normalize,
            learn_residual_step=learn_residual_step,
        )
        self.layer3 = self._make_layer(
            block_cls,
            num_blocks[2],
            320,
            bn_config,
            w_initializer,
            activation,
            stride=2,
            name="layer3",
            normalize=normalize,
            learn_residual_step=learn_residual_step,
            # drop_rate=drop_rate, drop_block=True, block_size=dropblock_size
        )
        self.layer4 = self._make_layer(
            block_cls,
            num_blocks[3],
            640,
            bn_config,
            w_initializer,
            activation,
            stride=2,
            name="layer4",
            normalize=normalize,
            learn_residual_step=learn_residual_step,
            # drop_rate=drop_rate, drop_block=True, block_size=dropblock_size
        )

    def _make_layer(
        self,
        block_cls,
        num_blocks,
        planes,
        bn_config,
        w_initializer,
        activation,
        stride=1,
        normalize="bn",
        learn_residual_step=False,
        name=None,
    ):
        use_projection = stride != 1 or self.inplanes != planes * block_cls.expansion
        self.inplanes = planes * block_cls.expansion
        return BlockGroup(
            block_cls,
            planes,
            num_blocks,
            stride,
            bn_config,
            w_initializer,
            activation,
            use_projection,
            normalize=normalize,
            learn_residual_step=learn_residual_step,
            name=name,
        )

    def __call__(self, x, mask=None, phase="all", training=None, test_local_stats=False, is_feat=False):
        assert phase in ["all", "encoder", "adaptation"]

        if phase in ["all", "encoder"]:

            x = self.layer1(x, training, test_local_stats)
            f0 = x
            x = self.layer2(x, training, test_local_stats)
            f1 = x
            x = self.layer3(x, training, test_local_stats)
            f2 = x
            x = self.layer4(x, training, test_local_stats)
            f3 = x
            if self.avg_pool:
                x = hk.avg_pool(x, jax.numpy.array([1, 5, 5, 1]), 1, padding="VALID", channel_axis=3)
                x = hk.Reshape((640,))(x)
            else:
                x = hk.Reshape((-1,))(x)
       
        if phase in ["all", "adaptation"]:
            x = hk.Linear(
                self.output_size,
                w_init=kaiming_normal,
                name="classifier",
            )(x)
        else:
            x = (x,)

        return x


def resnet12(*args, **kwargs):
    return ResNet(*args, **kwargs, block_cls=BasicBlock, num_blocks=[1, 1, 1, 1])