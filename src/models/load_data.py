import torchvision
from robustbench.utils import load_model as rb_load_model

from model_dataset import MODEL_NORMS, MODEL_DATASET


def load_dataset(dataset_name='cifar10'):
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('./models/data',
                                             train=False,
                                             download=True,
                                             transform=torchvision.transforms.ToTensor())
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('./models/data',
                                               train=False,
                                               download=True,
                                               transform=torchvision.transforms.ToTensor())

    return dataset


def load_model(model_name, dataset_name, norm='inf'):
    if norm not in MODEL_NORMS:
        norm = 'inf'

    norm_name = f'L{norm}'

    try:
        model = rb_load_model(
            model_dir="./models/pretrained",
            model_name=model_name,
            dataset=dataset_name,
            norm=norm_name
        )
    except KeyError:
        model = rb_load_model(
            model_dir="./models/pretrained",
            model_name=MODEL_DATASET[0]['model_name'],
            dataset=MODEL_DATASET[0]['datasets'][0],
            norm='Linf'
        )

    return model


def load_data(model_id=8, dataset_id=0, norm='inf'):
    """
    Load model and dataset (default: Gowal2021Improving_R18_ddpm_100m, CIFAR10)
    """
    model_id=int(model_id)
    dataset_id = int(dataset_id)
    model_id = 0 if model_id > len(MODEL_DATASET) else model_id
    dataset_id = 0 if dataset_id > len(MODEL_DATASET[model_id]['datasets']) else dataset_id

    model_name = MODEL_DATASET[model_id]['model_name']
    dataset_name = MODEL_DATASET[model_id]['datasets'][dataset_id]

    model = load_model(model_name, dataset_name, norm)
    dataset = load_dataset(dataset_name)

    return model, dataset
