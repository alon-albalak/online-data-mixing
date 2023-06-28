from typing import List, Optional
import numpy as np
import math
import torch

class BaseDataSamplingWeight:
    """
    Simple class that stores the weighting scheme for domains within the data.
    This class will maintain stationary weights, but can be subclassed to
    implement dynamic weighting schemes.
    """
    def __init__(self,
        weights: List[float] = None,
    ):
        self.weights = weights

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def __call__(self):
        return self.weights
    
    def update(self):
        pass
    
class DynamicDataSamplingWeight(BaseDataSamplingWeight):
    """
    Dynamic data sampling weighter that updates the weights based on the
    provided update method.
    """
    def __init__(self,
        dataset_names: List[str],
        weights: List[float] = None,
        warmup_steps: Optional[int] = 0,
        update_frequency: Optional[int] = 0,
        update_method: Optional[str] = None,
    ):
        super().__init__(weights=weights)
        self.dataset_names = dataset_names
        self.update_scheduler = UpdateScheduler(
            warmup_steps=warmup_steps,
            update_frequency=update_frequency,
        )
        self.update_method = update_method
        self.weight_updater = get_weight_updater(update_method, dataset_names, weights)
        
    def update(self, iteration: int, **kwargs):
        if self.update_scheduler.requires_update(iteration):
            self.weights = self.weight_updater.update(iteration=iteration, **kwargs)

class UpdateScheduler:
    def __init__(self, warmup_steps: int, update_frequency: int):
        self.warmup_steps = warmup_steps
        self.update_frequency = update_frequency

    def requires_update(self, iteration: int) -> bool:
        return iteration > self.warmup_steps and iteration % self.update_frequency == 0

class Exp3WeightUpdater:
    def __init__(
            self,
            dataset_names: List[str],
            weights: List[float],
            ):
        self.dataset_names = dataset_names
        self.dataset_map = {name: i for i, name in enumerate(dataset_names)}
        self.num_datasets = len(dataset_names)
        self.weights = weights
        self._cumulative_estimated_reward = {name: 0.0 for name in dataset_names}
        total_weights = np.sum(weights)
        self._probabilities = {name: weight/total_weights for name, weight in zip(dataset_names, weights)}
        self.eps = 1/self.num_datasets
        self.prev_eps = None

    def update(self, dataset_name: str, reward: float, iteration: int) -> List[float]:
        """
        Updates the weights based on the provided reward.
        """

        loss = reward
        # scale reward
        # reward = max(math.log(reward), 0)
        reward = reward/10

        # print(f"Rank: {torch.distributed.get_rank()} -- dataset_name: {dataset_name} -- reward: {reward}"
        #       f" -- eps {self.eps} -- prev_eps {self.prev_eps}")

        # update cumulative estimated reward
        self._cumulative_estimated_reward[dataset_name] += reward/self._probabilities[dataset_name]

        # print(f"Rank: {torch.distributed.get_rank()} -- cumulative_estimated_reward {self._cumulative_estimated_reward}")

        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/self.num_datasets, math.sqrt(math.log(self.num_datasets)/(self.num_datasets*iteration)))

        # print(f"Rank: {torch.distributed.get_rank()} -- eps {self.eps} -- prev_eps {self.prev_eps}")

        # calculate scaling factor
        total_estimated_rewards = sum([math.exp(r*self.prev_eps) for r in self._cumulative_estimated_reward.values()])
        scaling_factor = (1-self.num_datasets*self.eps)/total_estimated_rewards

        # print(f"Rank: {torch.distributed.get_rank()} -- total_estimated_rewards {total_estimated_rewards} -- scaling_factor {scaling_factor}")

        # update weights
        for name in self.dataset_names:
            self.weights[self.dataset_map[name]] = math.exp(self._cumulative_estimated_reward[name]*self.prev_eps)*scaling_factor + self.eps

        # print(f"Rank: {torch.distributed.get_rank()} -- cumulative_estimated_reward {self._cumulative_estimated_reward}")

        # update probabilities
        total_weights = sum(self.weights)
        for name in self.dataset_names:
            self._probabilities[name] = self.weights[self.dataset_map[name]]/total_weights

        print(f"Rank: {torch.distributed.get_rank()} -- loss {loss} -- reward: {reward} -- eps {self.eps} -- scaling_factor {scaling_factor}"
              f" -- cumulative_estimated_reward {self._cumulative_estimated_reward} -- weights {self.weights} -- probabilities {self._probabilities}"
              f" -- total_weights {total_weights}"
              )

        return list(self._probabilities.values())

        
def get_weight_updater(update_method: str, dataset_names, weights):
    if update_method == "exp3":
        return Exp3WeightUpdater(dataset_names=dataset_names, weights=weights)


def get_data_sampling_weighter(
        dataset_names: List[str],
        weights: List[float],
        warmup_steps: Optional[int] = 0,
        update_frequency: Optional[int] = 0,
        update_method: Optional[str] = None):
    """
    Returns a data sampling weighter based on the provided arguments.
    """
    if update_method is None:
        assert(warmup_steps==0 and update_frequency==0), "Must provide update method if warmup steps or update frequency are provided."
        return BaseDataSamplingWeight(weights=weights)
    
    return DynamicDataSamplingWeight(
        dataset_names=dataset_names,
        weights=weights,
        warmup_steps=warmup_steps,
        update_frequency=update_frequency,
        update_method=update_method,
    )