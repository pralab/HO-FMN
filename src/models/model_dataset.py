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
        }

}

MODEL_NORMS = [0, 1, 2, 'inf']

'''--------------------------------------------------------------------------------------------------

[CIFAR-10 Linf, eps=8/255]
Wang2023Better_WRN-70-16             1° standard acc = 93.25% / AA acc = 70.69%
Wang2023Better_WRN-28-10             2° standard acc = 92.44% / AA acc = 67.31%
Gowal2021Improving_70_16_ddpm_100m   4° standard acc = 88.74% / AA acc = 66.11%  - NeurIPS 2021
Rebuffi2021Fixing_106_16_cutmix_ddpm 7° standard acc = 88.50% / AA acc = 64.64%  - NeurIPS 2021
Gowal2021Improving_28_10_ddpm_100m   10° standard acc = 87.50% / AA acc = 63.44% - NeurIPS 2021
Pang2022Robustness_WRN70_16          11° standard acc = 89.01% / AA acc = 63.35% - ICML 2022
Sehwag2021Proxy_ResNest152           13° standard acc = 87.30% / AA acc = 62.79% - ICLR 2022
Pang2022Robustness_WRN28_10          18° standard acc = 88.61% / AA acc = 61.04% - ICML 2022
Gowal2021Improving_R18_ddpm_100m     28° standard acc = 87.35% / AA acc = 58.50% - NeurIPS 2021

'''
'''

[CIFAR-100 Linf, eps=8/255]

Wang2023Better_WRN-70-16            1°  standard acc = % / AA acc = %
Wang2023Better_WRN-28-10            2°  standard acc = % / AA acc = %
Pang2022Robustness_WRN70_16         7°  standard acc = % / AA acc = %
Pang2022Robustness_WRN28_10         14° standard acc = % / AA acc = %

'''
