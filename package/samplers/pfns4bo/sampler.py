from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any
from typing import cast
import warnings

import numpy as np
import optuna._gp.search_space as gp_search_space
from optuna._gp.search_space import _sample_normalized_params
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState
import torch

from pfns4bo import utils
from pfns4bo.model import bar_distribution
from pfns4bo.model.bar_distribution import BarDistributionConfig
from pfns4bo.model.encoders import EncoderConfig
from pfns4bo.priors import Batch
from pfns4bo.priors.prior import AdhocPriorConfig
from pfns4bo.scripts.acquisition_functions import optimize_acq_w_lbfgs
from pfns4bo.train import BatchShapeSamplerConfig
from pfns4bo.train import MainConfig
from pfns4bo.train import OptimizerConfig
from pfns4bo.train import train
from pfns4bo.train import TransformerConfig


hps = None
num_features = 1
max_dataset_size = 20


# in our convention we name the `num_datasets` -> `batch_size`, and the `num_points_in_each_dataset` -> `seq_len`


def get_batch_for_ridge_regression(
    batch_size: int = 2,
    seq_len: int = 100,
    num_features: int = 1,
    hyperparameters: dict | None = None,
    device: str = "cpu",
) -> Batch:
    if hyperparameters is None:
        hyperparameters = {"a": 0.1, "b": 1.0}
    ws = torch.distributions.Normal(torch.zeros(num_features + 1), hyperparameters["b"]).sample(
        (batch_size,)
    )

    xs = torch.rand(batch_size, seq_len, num_features)
    ys = torch.distributions.Normal(
        torch.einsum("nmf, nf -> nm", torch.cat([xs, torch.ones(batch_size, seq_len, 1)], 2), ws),
        hyperparameters["a"],
    ).sample()[..., None]

    # get_batch functions return two different ys, let's come back to this later, though.
    return Batch(x=xs.to(device), y=ys.to(device), target_y=ys.to(device))


def train_a_pfn(
    get_batch_function: Any,
    epochs: int = 10,
    num_features: int = num_features,
    max_dataset_size: int = max_dataset_size,
    hps: dict | None = hps,
    batch_size: int = 256,
    steps_per_epoch: int = 100,
) -> Any:
    # define a bar distribution (riemann distribution) criterion with 1000 bars
    ys = get_batch_function(100000, 20, num_features, hyperparameters=hps).target_y
    # we define our bar distribution adaptively with respect to the above sample of target ys from our prior
    borders = bar_distribution.get_bucket_borders(num_outputs=1_000, ys=ys).tolist()

    config = MainConfig(
        priors=[
            AdhocPriorConfig(
                get_batch_methods=[get_batch_function],
                prior_kwargs={"num_features": num_features, "hyperparameters": hps},
            )
        ],
        optimizer=OptimizerConfig("adamw", lr=0.0003),
        model=TransformerConfig(
            criterion=BarDistributionConfig(full_support=True, borders=borders),
            emsize=512,
            nhead=8,
            nhid=1024,
            nlayers=6,
            features_per_group=1,
            attention_between_features=False,
            # The encoder config ensures the uniform inputs between 0 and 1 have mean 0 and var 1
            encoder=EncoderConfig(
                constant_normalization_mean=0.5,
                constant_normalization_std=math.sqrt(1 / 12),
            ),
        ),
        batch_shape_sampler=BatchShapeSamplerConfig(
            batch_size=batch_size,
            max_seq_len=max_dataset_size,
            min_num_features=num_features,
            max_num_features=num_features,
        ),
        epochs=epochs,
        warmup_epochs=epochs // 4,
        steps_per_epoch=steps_per_epoch,
        num_workers=0,
    )
    train_result = train(config, device="cpu", reusable_config=False)
    return train_result


class PFNs4BOSampler(BaseSampler):
    """A sampler based on the Prior-data Fitted Networks (PFNs) as the surrogate model.

    This sampler is based on the PFNs, which is a neural network-based surrogate model.

    .. note::
        The default prior argument is ``"hebo"``. This trains the PFNs model in the
        init of the sampler. If you want to use a pre-trained model, you can download
        the model checkpoint from the following link:
        https://github.com/automl/PFNs4BO/tree/main/pfns4bo/final_models
        and load it using the following code:

        .. code-block:: python
            import torch

            model = torch.load("PATH/TO/MODEL.pt")
            sampler = PFNs4BOSampler(prior=model)

    .. note::
        The performance of PFNs4BO with the HEBO+ prior is maximized with the number of
        trials smaller than 100 or 200 in most cases. If you have a large number of trials,
        it is recommended to change the sampler to a random sampler or etc after a certain
        number of trials.

    Args:
        prior:
            A string or a torch.nn.Module object. If a string, it should be one of the following:

            - ``"vanilla gp"``: A vanilla GP model.
            - ``"hebo"``: A model based on the HEBO+ algorithm.

            If a torch.nn.Module object, it should be a trained model.
        model_path:
            A file path to save the trained model. If None, the model will not be saved.
        seed:
            Seed for random number generator.
        independent_sampler:
            A sampler instance for independent sampling. If None, :class:`~optuna.samplers.RandomSampler`
            is used.
        n_startup_trials:
            The number of initial trials that are used to fit the model.
        num_grad_steps:
            The number of gradient steps for optimization.
        num_candidates:
            The number of candidates for optimization.
        pre_sample_size:
            The number of samples for pre-sampling.
        acquisition_function_type:
            The type of acquisition function. It should be one of the following:

            - ``"ei"``: Expected improvement.
            - ``"pi"``: Probability of improvement.
            - ``"ucb"``: Upper confidence bound.
            - ``"ei_or_rand"``: Expected improvement mixed with random sampling.
            - ``"mean"``: Mean of the model.
    """

    def __init__(
        self,
        *,
        prior: str | torch.nn.Module = "hebo",
        model_path: str | None = None,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        num_grad_steps: int = 15_000,
        num_candidates: int = 100,
        pre_sample_size: int = 100_000,
        acquisition_function_type: str = "ei",
    ) -> None:
        self._num_grad_steps = num_grad_steps
        self._num_candidates = num_candidates
        self._pre_sample_size = pre_sample_size
        self._acquisition_function_type = acquisition_function_type

        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)
        self._intersection_search_space = IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials

        self._device = utils.default_device

        if isinstance(prior, torch.nn.Module):
            trained_model = prior
        elif prior == "vanilla gp":
            _, _, trained_model, _ = train()
        elif prior == "hebo":
            _, _, trained_model, _ = train_a_pfn(get_batch_for_ridge_regression)
        else:
            raise ValueError("You should specify `prior` as 'vanilla gp', 'hebo', or a model.")

        self._model = trained_model
        self._model.eval()

        if model_path is not None:
            torch.save(trained_model, model_path)

    def sample_relative(
        self, study: Study, trial: Trial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        (
            internal_search_space,
            normalized_params,
        ) = gp_search_space.get_search_space_and_normalized_params(trials, search_space)

        _sign = -1.0 if study.direction == StudyDirection.MINIMIZE else 1.0
        score_vals = np.array([_sign * cast(float, trial.value) for trial in trials])

        if np.any(~np.isfinite(score_vals)):
            warnings.warn(
                "This sampler cannot handle infinite values. "
                "We clamp those values to worst/best finite value."
            )

            finite_score_vals = score_vals[np.isfinite(score_vals)]
            best_finite_score = np.max(finite_score_vals, initial=0.0)
            worst_finite_score = np.min(finite_score_vals, initial=0.0)

            score_vals = np.clip(score_vals, worst_finite_score, best_finite_score)

        standarized_score_vals = (score_vals - score_vals.mean()) / max(1e-10, score_vals.std())

        def rand_sample_func(n: int) -> torch.Tensor:
            xs = _sample_normalized_params(n, internal_search_space, None)
            ret = torch.from_numpy(xs).to(torch.float32).to(self._device)
            return ret

        known_x = torch.from_numpy(normalized_params).to(torch.float32).to(self._device)
        known_y = torch.from_numpy(standarized_score_vals).to(torch.float32).to(self._device)

        with torch.enable_grad():
            _, x_options, eis, _, _ = optimize_acq_w_lbfgs(
                model=self._model,
                known_x=known_x,
                known_y=known_y,
                num_grad_steps=self._num_grad_steps,
                num_candidates=self._num_candidates,
                pre_sample_size=self._pre_sample_size,
                device=self._device,
                rand_sample_func=rand_sample_func,
                apply_power_transform=True,
                acq_function=self._acquisition_function_type,
            )

        normalized_param = x_options[torch.argmax(eis)]
        return gp_search_space.get_unnormalized_param(search_space, normalized_param)

    def infer_relative_search_space(
        self, study: Study, trial: Trial
    ) -> dict[str, BaseDistribution]:
        search_space = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def sample_independent(
        self,
        study: Study,
        trial: Trial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
