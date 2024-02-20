import torchvision
import torchvision.transforms as transforms

from robustbench.utils import load_model as rb_load_model

MODEL_DATASET = {
    0: {
        'model_name': 'Wang2023Better_WRN-70-16',
        'datasets': ['cifar10']
        },
    1: {'model_name': 'Wang2023Better_WRN-28-10',
        'datasets': ['cifar10']
        },
    2: {'model_name': 'Gowal2021Improving_70_16_ddpm_100m',
        'datasets': ['cifar10']
        },
    3: {'model_name': 'Rebuffi2021Fixing_106_16_cutmix_ddpm',
        'datasets': ['cifar10']
        },
    4: {'model_name': 'Gowal2021Improving_28_10_ddpm_100m',
        'datasets': ['cifar10']
        },
    5: {'model_name': 'Pang2022Robustness_WRN70_16',
        'datasets': ['cifar10']
        },
    6: {'model_name': 'Sehwag2021Proxy_ResNest152',
        'datasets': ['cifar10']
        },
    7: {'model_name': 'Pang2022Robustness_WRN28_10',
        'datasets': ['cifar10']
        },
    8: {'model_name': 'Gowal2021Improving_R18_ddpm_100m',
        'datasets': ['cifar10']
        },
    9: {'model_name': 'Rade2021Helper_R18_ddpm',
        'datasets': ['cifar10']
        },
    10: {'model_name': 'Sehwag2021Proxy_R18',
        'datasets': ['cifar10']
        },
    11: {'model_name': 'Rebuffi2021Fixing_R18_ddpm',
        'datasets': ['cifar10']
        }
}

MODEL_NORMS = ["L0", "L1", "L2", "Linf"]


def load_dataset(dataset_name='cifar10'):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('../models/data',
                                             train=False,
                                             download=True,
                                             transform=transform)
    else:
        dataset = torchvision.datasets.CIFAR10('../models/data',
                                               train=False,
                                               download=True,
                                               transform=transform)

    return dataset


def load_model(model_name, dataset_name, norm="Linf"):
    if norm not in MODEL_NORMS:
        norm = "Linf"

    try:
        model = rb_load_model(
            model_dir="../models/pretrained",
            model_name=model_name,
            dataset=dataset_name,
            norm=norm
        )
    except KeyError:
        model = rb_load_model(
            model_dir="../models/pretrained",
            model_name=MODEL_DATASET[0]['model_name'],
            dataset=MODEL_DATASET[0]['datasets'][0],
            norm='Linf'
        )

    return model


def load_data(model_id=0, dataset_id=0, norm='inf'):
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

    return model, dataset, model_name, dataset_name


def print_models_info():
    print("Model ID\tName")
    for model_id in MODEL_DATASET:
        print(f"{model_id}\t\t\t{MODEL_DATASET[model_id]['model_name']}")
