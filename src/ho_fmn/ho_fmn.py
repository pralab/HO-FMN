import os
from datetime import datetime
from typing import Literal, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.base import ModelBridge

from src.attacks.fmn import FMN
from .search_space import OPTIMIZER_PARAMS, SCHEDULER_PARAMS


class HOFMN:
    def __init__(self,
                 model: nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 exp_path: Optional[str] = '../experiments',
                 exp_name: Optional[Union[str, None]] = None,
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
        """
        HO-FMN: Hyperparameter Optimization for Fast Minimum-Norm Attacks.

        Parameters:
            model (nn.Module): Model (torch) to be attacked.
            dataloader = (torch.utils.data.DataLoader): Dataloader for loading images, labels.
            exp_path = (str): Where to store the experiment data. Default is './Experiments'.
            batch_size (int): Size of the batches for optimization. Default is 32.
            loss (Literal['LL', 'CE', 'DLR']): Type of loss function to use. Default is 'LL'.
            optimizer (Literal['SGD', 'Adam', 'Adamax']): Type of optimizer to use. Default is 'SGD'.
            scheduler (Literal['CALR', 'RLROP', None]): Type of learning rate scheduler to use. Default is 'CALR'.
            steps (int): Number of attack steps. Default is 100.
            norm (float): Norm constraint for the attack. Default is torch.inf.
            trials (int): Number of trials for the Hyperparameter optimization. Default is 32.
            fixed_batch (bool): Whether to use a fixed batch for optimization. Default is False.
            verbose (bool): Whether to print verbose information during optimization. Default is False.
            device (torch.device): Device to run the optimization on. Default is torch.device('cpu').
        """
        self.model = model
        self.dataloader = dataloader
        self.exp_path = exp_path
        self.batch_size = self.dataloader.batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.steps = steps
        self.norm = norm
        self.trials = trials
        self.fixed_batch = fixed_batch
        self.verbose = verbose
        self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()

        # Retrieve optimizer and scheduler params
        self.opt_params = OPTIMIZER_PARAMS.get(optimizer, 'SGD').copy()
        self.sch_params = None
        if scheduler is not None:
            self.sch_params = SCHEDULER_PARAMS.get(scheduler, 'CALR').copy()
            if 'T_max' in self.sch_params:
                self.sch_params['T_max'] = self.sch_params['T_max'](steps)
            if 'batch_size' in self.sch_params:
                self.sch_params['batch_size'] = self.sch_params['batch_size'](self.batch_size)

        # Create experiment name and folder
        self.exp_path = exp_path

        if exp_name is None:
            self.exp_name = f'{self.batch_size}_{self.steps}_{self.trials}_{self.optimizer}_{self.scheduler}_{self.loss}'
        else:
            self.exp_name = exp_name

        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path, exist_ok=True)

        self.ax_client = None

    def get_tuning_model(self) -> ModelBridge:
        """
        Get the predictive model obtained from the tuning process.

        Parameters:
            None.
        Returns:
            model (ModelBridge): The predictive model.
        """
        model = None
        if self.ax_client is not None:
            model = self.ax_client.generation_strategy.model

        return model

    def parametrization_to_configs(self,
                                   parametrization: dict,
                                   steps: Optional[int] = None,
                                   batch_size: Optional[int] = None,
                                   ) -> tuple[dict, dict]:
        optimizer_config = {k: parametrization[k] for k in set(self.opt_params)}
        scheduler_config = {k: parametrization[k] for k in set(self.sch_params)} if self.scheduler else None

        if steps is not None and 'T_max' in scheduler_config:
            scheduler_config['T_max'] = steps
        if batch_size is not None and 'batch_size' in scheduler_config:
            scheduler_config['batch_size'] = batch_size

        return optimizer_config, scheduler_config

    def _evaluate(self, parametrization: dict, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Evaluate FMN with a given parametrization.

        Parameters:
            parametrization (dict): A dictionary containing the parameters of the FMN.
            images (torch.Tensor): Input images to evaluate.
            labels (torch.Tensor): Ground truth labels corresponding to the input images.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        optimizer_config, scheduler_config = self.parametrization_to_configs(parametrization)

        attack = FMN(
            model=self.model,
            steps=self.steps,
            loss=self.loss,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            norm=self.norm,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            device=self.device
        )
        best_adv = attack.forward(images=images, labels=labels)
        best_adv = best_adv.to(self.device)

        best_distance = torch.linalg.norm((best_adv - images).data.flatten(1), dim=1, ord=self.norm).clone().detach()
        best_distance = torch.where(best_distance > 0, best_distance, torch.tensor(float('inf')))
        evaluation = {'distance': (best_distance.median().item(), 0.0)}
        
        return evaluation

    def tune(self) -> dict:
        """
        Tune the hyperparameters of the FMN attack given a model, a loss function, an optimizer and a scheduler.

        Parameters:
            None.
        Returns: A dictionary containing the best parameters found during the HO.
        """

        # Create Ax Experiment
        if self.verbose:
            print("\t[Tuning] Creating the Ax client and experiment...")
        self.ax_client = AxClient()

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
        self.ax_client.create_experiment(
            name=self.exp_name,
            parameters=params,
            objectives=objectives
        )

        if self.verbose:
            print("\t[Tuning] Starting the Hyperparameter Optimization...")

        images, labels = next(iter(self.dataloader))
        for i in range(self.trials):
            if self.verbose:
                print(f"\t[Tuning] Running trial {i}")

            if not self.fixed_batch:
                images, labels = next(iter(self.dataloader))
            images, labels = images.to(self.device), labels.to(self.device)

            parameters, trial_index = self.ax_client.get_next_trial()
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=self._evaluate(parameters, images, labels)
            )

        if self.verbose:
            print("\t[Tuning] Finished the Hyperparameter Optimization; printing the trials"
                  "list and best parameters...")
            print(self.ax_client.get_trials_data_frame())

        best_parameters, values = self.ax_client.get_best_parameters()

        if self.verbose:
            print("\t[Tuning] Best parameters: ", best_parameters)
            print("\t[Tuning] Saving the experiment data...")

        current_date = datetime.now()
        formatted_date = current_date.strftime("%d%m%y%H")
        exp_json_path = os.path.join(self.exp_path, f'{formatted_date}_{self.exp_name}.json')
        self.ax_client.save_to_json_file(filepath=exp_json_path)

        return best_parameters
