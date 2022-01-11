import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset


def get_dataloader(
        dataset_name, path_to_data: str, cid: str, is_train: bool, batch_size: int, workers: int
):
    """Generates trainset/valset object and returns appropiate dataloader."""
    transform_train, transform_test = cifar10Transformation() if dataset_name == 'CIFAR-10' else cifar100Transformation()
    partition = "train" if is_train else "test"
    dataset = TorchVision_FL(Path(path_to_data) / cid / (partition + ".pt"),
                             transform=transform_train if is_train else transform_test)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def create_lda_partitions(dataset, dirichlet_dist):
    img_ids, labels = dataset
    min_size = 0
    min_require_size = 10
    num_classes = len(dirichlet_dist)
    num_parties = len(dirichlet_dist[0])
    N = labels.shape[0]
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_parties)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = dirichlet_dist[k]
            proportions = np.array([p * (len(idx_j) < N / num_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    partitions = []
    for idx_j in idx_batch:
        partitions.append((img_ids[idx_j], labels[idx_j]))
    return partitions


def do_fl_partitioning(path_to_dataset, experiment_id, pool_size, dirichlet_dist):
    """Torchvision (e.g. CIFAR-10) datasets using LDA."""

    splits_dir = path_to_dataset / experiment_id / "federated"

    train_images, train_labels = torch.load(path_to_dataset / "training.pt")
    train_idx = np.array(range(len(train_images)))
    train_dataset = [train_idx, train_labels]

    train_partitions = create_lda_partitions(train_dataset, dirichlet_dist)

    test_images, test_labels = torch.load(path_to_dataset / "test.pt")
    print(len(test_images), len(test_labels))
    test_idx = np.array(range(len(test_images)))
    test_dataset = [test_idx, test_labels]

    test_partitions = create_lda_partitions(test_dataset, dirichlet_dist)

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(pool_size):
        # create dir
        if not (splits_dir / str(p)).exists():
            Path.mkdir(splits_dir / str(p))

        train_labels = train_partitions[p][1]
        train_image_idx = train_partitions[p][0]
        train_imgs = train_images[train_image_idx]

        test_labels = test_partitions[p][1]
        test_image_idx = test_partitions[p][0]
        test_imgs = test_images[test_image_idx]

        with open(splits_dir / str(p) / "train.pt", "wb") as f:
            torch.save([train_imgs, train_labels], f)
        with open(splits_dir / str(p) / "test.pt", "wb") as f:
            torch.save([test_imgs, test_labels], f)

    return splits_dir


def cifar10Transformation():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    # data prep for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return transform_train, transform_test


def cifar100Transformation():
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])
    # data prep for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return transform_train, transform_test


class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
            self,
            path_to_data=None,
            data=None,
            targets=None,
            transform: Optional[Callable] = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def getCIFAR100(path_to_data="./data"):
    """Downloads CIFAR100 dataset and generates unified training and test sets (they will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train and test sets
    train_set = datasets.CIFAR100(root=path_to_data, train=True, download=True)
    test_set = datasets.CIFAR100(
        root=path_to_data, train=False, download=False
    )

    # fuse all data splits 
    data_loc = Path(path_to_data) / "cifar-100-batches-py"
    training_data = data_loc / "training.pt"
    test_data = data_loc / "test.pt"
    print("Generating unified CIFAR dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)
    torch.save([test_set.data, np.array(train_set.targets)], test_data)

    test_set = datasets.CIFAR100(
        root=path_to_data, train=False, transform=cifar100Transformation()[1]
    )
    return training_data, test_data, test_set


def getCIFAR10(path_to_data="./data"):
    """Downloads CIFAR10 dataset and generates unified training and test sets (they will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train and test sets
    train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)
    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, download=False
    )

    # fuse all data splits 
    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"
    test_data = data_loc / "test.pt"
    print("Generating unified CIFAR dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)
    torch.save([test_set.data, np.array(test_set.targets)], test_data)

    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, transform=cifar10Transformation()[1]
    )
    return data_loc, test_set
