from numpy import load
from jax.random import split
import data.augmentations as augmentations


def augment(rng, imgs, color_jitter_prob=1.0, out_size=84):
    rng_crop, rng_color, rng_flip = split(rng, 3)
    # print("crop")
    imgs = augmentations.random_crop(imgs, rng_crop, out_size, ((8, 8), (8, 8), (0, 0)))
    # print("color")
    # imgs = augmentations.color_transform(
    #     imgs,
    #     rng_color,
    #     brightness=0.4,
    #     contrast=0.4,
    #     saturation=0.4,
    #     hue=0.0,
    #     color_jitter_prob=color_jitter_prob,
    #     to_grayscale_prob=0.0,
    # )
    # print("flip")
    # imgs = augmentations.random_flip(imgs, rng_flip)
    # imgs = normalize_fn(imgs)
    return imgs


class TensorDataset:
    def __init__(self, *tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple((t[index] for t in self.tensors))

    def __len__(self):
        return len(self.tensors[0])


class ImageDataset:
    def __init__(self, fp, norm_0_1=True):
        data = load(fp)
        self.images = data["images"]
        self.targets = data["targets"]

        self._mean_255 = data["mean"]
        self._std_255 = data["std"]
        self._mean_0_1 = data["mean_0_1"]
        self._std_0_1 = data["std_0_1"]

        if norm_0_1:
            self.mean = self._mean_0_1
            self.std = self._std_0_1
        else:
            self.mean = self._mean_255
            self.std = self._std_255

        assert len(self.images) == len(self.targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]