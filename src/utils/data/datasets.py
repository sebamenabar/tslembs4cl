import numpy as onp


class Subset:
    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indexes = indexes
        assert onp.max(indexes) <= len(dataset)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return self.dataset[self.indexes[index]]


class SimpleDataset:
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def get_classes_subset(self, classes, sort=False):
        indexes = onp.isin(self.targets, classes)

        inputs = []
        targets = []

        if sort:
            classes = sorted(classes)

        for _class in classes:
            indexes = self.targets == _class
            assert (
                indexes.any()
            ), f"Class {_class} not in targets (all classes: {classes})"
            inputs.append(self.inputs[indexes])
            targets.append(self.targets[indexes])

        return SimpleDataset(
            onp.concatenate(inputs),
            onp.concatenate(targets),
        )


class ImageDataset(SimpleDataset):
    def __init__(self, fp, norm_0_1=True, transpose=True, no_stats_ok=False):
        data = onp.load(fp)
        images = data["images"]
        if transpose:
            images = images.transpose((0, 2, 3, 1))
        targets = data["targets"]

        _mean_255 = None
        _std_255 = None
        _mean_0_1 = None
        _std_0_1 = None

        try:
            _mean_255 = data["mean"]
            _std_255 = data["std"]
            _mean_0_1 = data["mean_0_1"]
            _std_0_1 = data["std_0_1"]
        except Exception as e:
            if no_stats_ok:
                pass
            else:
                raise (e)

        if norm_0_1:
            mean = _mean_0_1
            std = _std_0_1
        else:
            mean = _mean_255
            std = _std_255

        self.mean = mean
        self.std = std

        assert len(images) == len(targets)

        super().__init__(images, targets)

    def normalize(self, x):
        return (x - self.mean) / self.std