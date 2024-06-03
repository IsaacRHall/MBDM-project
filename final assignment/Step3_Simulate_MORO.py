import os
import pandas as pd
import random
from ema_workbench import (Model, Policy, ema_logging, MultiprocessingEvaluator, ScalarOutcome)
from ema_workbench.em_framework import samplers
from ema_workbench.em_framework.evaluators import BaseEvaluator
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)
import functools
import numpy as np
from problem_formulation import get_model_for_problem_formulation


import os
import pandas as pd
import random
from ema_workbench import (Model, Policy, ema_logging, MultiprocessingEvaluator, ScalarOutcome)
from ema_workbench.em_framework import samplers
from problem_formulation import get_model_for_problem_formulation


# Define the robustness functions
def robustness(direction, threshold, data):
    if direction == 'SMALLER':
        return np.sum(data <= threshold) / data.shape[0]
    else:
        return np.sum(data >= threshold) / data.shape[0]

sum_over = functools.partial(np.sum, axis=0)

def process_outcomes(outcomes):
    processed = {}
    for key, value in outcomes.items():
        if isinstance(value, np.ndarray):
            processed[key] = value.sum(axis=1) if value.ndim > 1 else value
        else:
            processed[key] = value
    return pd.DataFrame(processed)

if __name__ == "__main__":
    problem_formulation_id = 7

    # Set up logging and seed
    random.seed(1361)
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Initialize the dike model from the problem formulation
    dike_model, planning_steps = get_model_for_problem_formulation(problem_formulation_id)

    # Ensure the output directory exists
    output_dir = os.path.join('data', 'output_data')
    os.makedirs(output_dir, exist_ok=True)

    # Sample scenarios
    n_scenarios = 5
    scenarios = samplers.sample_uncertainties(dike_model, n_scenarios)

    # Define robustness functions for optimization
    max_damage = functools.partial(robustness, 'SMALLER', 1000000)
    max_investment = functools.partial(robustness, 'SMALLER', 10000000)
    max_deaths = functools.partial(robustness, 'SMALLER', 0.05)

    robustness_functions = [
        ScalarOutcome('Damage Robustness', kind=ScalarOutcome.MAXIMIZE,
                      variable_name='Combined_Expected Annual Damage', function=max_damage),
        ScalarOutcome('Investment Robustness', kind=ScalarOutcome.MAXIMIZE,
                      variable_name='Combined_Dike Investment Costs', function=max_investment),
        ScalarOutcome('Deaths Robustness', kind=ScalarOutcome.MAXIMIZE,
                      variable_name='Combined_Expected Number of Deaths', function=max_deaths)
    ]

    convergence_metrics = [HyperVolume(minimum=[0, 0, 0], maximum=[1400000000, 300000000, 1]), EpsilonProgress()]
    nfe = 700

    # Run optimization
    with MultiprocessingEvaluator(dike_model) as evaluator:
        archive, convergence = evaluator.robust_optimize(robustness_functions, scenarios, nfe=nfe,
                                                         convergence=convergence_metrics,
                                                         epsilons=[1e6, 1e6, 0.012]
                                                         )

    # Save results
    archive.to_csv(os.path.join(output_dir, 'optimization_archive.csv'))

    # Save convergence metrics
    convergence_df = pd.DataFrame({
        'nfe': convergence.nfe,
        'epsilon_progress': convergence.epsilon_progress,
        'hypervolume': convergence.hypervolume
    })
    convergence_df.to_csv(os.path.join(output_dir, 'convergence_metrics.csv'))

