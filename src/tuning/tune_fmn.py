import os

import torch
import ray
from ray import air
from ray.air import session
from ray import tune
from ray.tune.search.flaml import CFO
from ray.tune.schedulers import ASHAScheduler

from src.attacks.fmn_opt import FMNOpt
from src.tuning.search_space import OPTIMIZERS_SEARCH_TUNE, SCHEDULERS_SEARCH_TUNE
from src.tuning.tuning_resources import TUNING_RES


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"

def tune_attack(config, model, samples, labels, attack_params, epochs=5):
    for epoch in range(epochs):
        attack = FMNOpt(
            model=model,
            inputs=samples.clone(),
            labels=labels.clone(),
            norm=attack_params['norm'],
            steps=attack_params['steps'],
            optimizer=attack_params['optimizer'],
            scheduler=attack_params['scheduler'],
            optimizer_config=config['opt_s'],
            scheduler_config=config['sch_s'],
            device=device,
            logit_loss= True if attack_params['loss'] == 'LL' else False
        )

        distance, _ = attack.run()
        session.report({"distance": distance})


def tune_fmn(
        model,
        data_loader,
        optimizer,
        scheduler,
        batch,
        steps,
        num_samples,
        loss,
        epochs=1):
    """
    :param optimizer: ...
    :param scheduler: ...
    """

    # load search spaces
    optimizer_search = OPTIMIZERS_SEARCH_TUNE[optimizer]
    scheduler_search = SCHEDULERS_SEARCH_TUNE[scheduler]

    attack_params = {
        'batch': int(batch),
        'steps': int(steps),
        'norm': 'inf',
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss': loss
    }

    tune_config = {
        'num_samples': int(num_samples),
        'epochs': int(epochs)
    }

    inputs, labels = next(iter(data_loader))

    steps_keys = ['T_max', 'T_0', 'milestones']
    for key in steps_keys:
        if key in scheduler_search:
            scheduler_search[key] = scheduler_search[key](attack_params['steps'])
    search_space = {
        'opt_s': optimizer_search,
        'sch_s': scheduler_search
    }  

    tune_with_resources = tune.with_resources(
        tune.with_parameters(
            tune_attack,
            model=torch.nn.DataParallel(model).to(device),
            samples=inputs.to(device),
            labels=labels.to(device),
            attack_params=attack_params,
            epochs=tune_config['epochs']
        ),
        resources=TUNING_RES
    )

    tune_scheduler = ASHAScheduler(mode='min', metric='distance', grace_period=2)
    algo = CFO(metric='distance', mode='min')

    tuning_exp_name = f"{optimizer}_{scheduler}_{loss}"
    tuner = tune.Tuner(
        tune_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=tune_config['num_samples'],
            search_alg=algo,
            scheduler=tune_scheduler
        ),
        run_config=air.RunConfig(
            tuning_exp_name,
            local_dir='./tuning_data',
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=False,
                checkpoint_frequency=0,
                num_to_keep=None),
            log_to_file=False,
            verbose=1
        )
    )
    
    results = tuner.fit()
    ray.shutdown()

    # Checking best result and best config
    best_result = results.get_best_result(metric='distance', mode='min')
    best_config = best_result.config
    print(f"best_distance : {best_result.metrics['distance']}\n, best config : {best_config}\n")

    return best_config
