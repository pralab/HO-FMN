import os
from datetime import datetime
from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from fmn import FMN
from utils.model_data import load_data
from tuning.search_space import OPTIMIZER_PARAMS, SCHEDULER_PARAMS


class HOFMN:
    def __init__(self,
                 model_id: int = 8,
                 model_name: Optional[str] = None,
                 batch_size: int = 32,
                 loss: Literal['LL', 'CE', 'DLR'] = 'LL',
                 optimizer: Literal['SGD', 'Adam', 'Adamax'] = 'SGD',
                 scheduler: Literal['CALR', 'RLROP', None] = 'CALR',
                 steps: int = 100,
                 norm: float = torch.inf,
                 trials: int = 32,
                 fixed_batch: bool = False,
                 verbose: bool = False,
                 device: torch.device = torch.device('cpu')
                 ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.steps = steps
        self.norm = norm
        self.trials = trials
        self.fixed_batch = fixed_batch
        self.verbose = verbose
        self.device = device

        if self.verbose:
            print("\t[Tuning] Retrieving the model and the dataset...")
        model, dataset, model_name, dataset_name = load_data(model_id=model_id)
        model.eval()
        self.model = model.to(self.device)

        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Retrieve optimizer and scheduler params
        self.opt_params = OPTIMIZER_PARAMS[optimizer]
        self.sch_params = None
        if scheduler is not None:
            self.sch_params = SCHEDULER_PARAMS[scheduler]
            if 'T_max' in self.sch_params:
                self.sch_params['T_max'] = self.sch_params['T_max'](steps)
            if 'batch_size' in self.sch_params:
                self.sch_params['batch_size'] = self.sch_params['batch_size'](batch_size)

        # Create experiment name and folder
        self.experiment_name = f'{self.model_name}_{self.batch_size}_{self.steps}_{self.n_trials}_{self.optimizer}_{self.scheduler}_{self.loss}'
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name, exist_ok=True)

    def _evaluate(self, parametrization: dict, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Evaluate FMN with a given parametrization
        """
        optimizer_config = {k: parametrization[k] for k in set(self.opt_params)}
        scheduler_config = None
        if self.scheduler is not None:
            scheduler_config = {k: parametrization[k] for k in set(self.sch_params)}

        attack = FMN(
            model=self.model,
            steps=self.steps,
            loss=self.loss,
            device=self.device,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            norm=self.norm,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config
        )

        best_adv = attack.forward(images=images, labels=labels)

        best_distance = torch.linalg.norm((best_adv - images).data.flatten(1), dim=1, ord=self.norm).clone().detach().cpu()
        best_distance = torch.where(best_distance > 0, best_distance, torch.tensor(float('inf')))

        evaluation = {'distance': (best_distance, 0.0)}

        return evaluation


    def tune(self, dataloader: torch.utils.data.DataLoader) -> dict:
        # Create Ax Experiment
        print("\t[Tuning] Creating the Ax client and experiment...")
        ax_client = AxClient()

        # Defining the Search Space
        if self.scheduler is not None:
            params = list(self.opt_params.values()) + list(self.sch_params.values())
        else:
            params = list(self.opt_params.values())

        # Define the objective(s)
        objectives = {
            'distance': ObjectiveProperties(minimize=True, threshold=8/255*2)
        }

        # Create an experiment with required arguments: name, parameters, and objective_name.
        ax_client.create_experiment(
            name=self.experiment_name,
            parameters=params,
            objectives=objectives
        )

        print("\t[Tuning] Starting the Hyperparameters Optimization...")
        for i in range(self.trials):
            print(f"\t[Tuning] Running trial {i}")

            images, labels = next(iter(self.dataloader))
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=self._evaluate(parameters, images, labels)
            )

        print("\t[Tuning] Finished the Hyperparameters Optimization; printing the trials list and best parameters...")
        print(ax_client.get_trials_data_frame())

        best_parameters, values = ax_client.get_best_parameters()
        print("\t[Tuning] Best parameters: ", best_parameters)

        print("\t[Tuning] Saving the experiment data...")
        current_date = datetime.now()
        formatted_date = current_date.strftime("%d%m%y%H")
        exp_json_path = os.path.join(self.experiment_name, f'{formatted_date}_{self.experiment_name}.json')
        ax_client.save_to_json_file(filepath=exp_json_path)

        return best_parameters
