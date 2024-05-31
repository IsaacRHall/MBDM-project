import os
import pandas as pd
import random
from ema_workbench import (Model, Policy, ema_logging, MultiprocessingEvaluator, Samplers)
from problem_formulation import get_model_for_problem_formulation
import numpy as np

def get_do_nothing_dict(model):
    """
    Prepare a dictionary for a 'do nothing' policy.

    Parameters
    ----------
    model : Model
        The model for which the policy dictionary is being prepared.

    Returns
    -------
    dict
        A dictionary with all policy levers set to zero.
    """
    return {lever.name: 0 for lever in model.levers}



def process_outcomes(outcomes):
    """
    Process outcomes to handle multidimensional arrays and structure the DataFrame correctly.

    Parameters
    ----------
    outcomes : dict
        Dictionary of outcomes from the model.

    Returns
    -------
    pd.DataFrame
        A DataFrame with processed outcomes.
    """
    processed = {}
    for key, value in outcomes.items():
        if isinstance(value, np.ndarray):
            processed[key] = value.sum(axis=1) if value.ndim > 1 else value
        else:
            processed[key] = value

    return pd.DataFrame(processed)

if __name__ == "__main__":
    problem_formulation_id = 6

    # Set up logging and seed
    random.seed(1361)
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Initialize the dike model from the problem formulation
    dike_model, planning_steps = get_model_for_problem_formulation(problem_formulation_id)

    # Ensure the output directory exists
    output_dir = os.path.join('data', 'output_data')
    os.makedirs(output_dir, exist_ok=True)

    # Random Policy Analysis
    with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
        random_experiments, random_outcomes = evaluator.perform_experiments(
            scenarios=1000, policies=500
        )
    random_experiments.to_csv(os.path.join(output_dir, 'random_experiments_combined.csv'))
    random_outcomes_df = process_outcomes(random_outcomes)
    random_outcomes_df['policy'] = random_experiments['policy']
    random_outcomes_df.to_csv(os.path.join(output_dir, 'random_outcomes_combined.csv'))

    # Sobol Sensitivity Analysis
    base_case_policy = [Policy("Do Nothing Policy", **get_do_nothing_dict(dike_model))]
    with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
        sobol_experiments, sobol_outcomes = evaluator.perform_experiments(
            scenarios=1024, policies=base_case_policy, uncertainty_sampling=Samplers.SOBOL
        )
    sobol_experiments.to_csv(os.path.join(output_dir, 'sobol_experiments_combined.csv'))
    sobol_outcomes_df = process_outcomes(sobol_outcomes)
    sobol_outcomes_df.to_csv(os.path.join(output_dir, 'sobol_outcomes_combined.csv'))

    # No Policy Analysis
    with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
        no_policy_experiments, no_policy_outcomes = evaluator.perform_experiments(
            scenarios=1000000, policies=base_case_policy
        )
    no_policy_experiments.to_csv(os.path.join(output_dir, 'no_policy_experiments_combined.csv'))
    no_policy_outcomes_df = process_outcomes(no_policy_outcomes)
    no_policy_outcomes_df['policy'] = no_policy_experiments['policy']
    no_policy_outcomes_df.to_csv(os.path.join(output_dir, 'no_policy_outcomes_combined.csv'))


# def get_do_nothing_dict(model):
#     """
#     Prepare a dictionary for a 'do nothing' policy.
#
#     Parameters
#     ----------
#     model : Model
#         The model for which the policy dictionary is being prepared.
#
#     Returns
#     -------
#     dict
#         A dictionary with all policy levers set to zero.
#     """
#     return {lever.name: 0 for lever in model.levers}
#
#
# if __name__ == "__main__":
#     # Set up logging and seed
#     random.seed(1361)
#     ema_logging.log_to_stderr(ema_logging.INFO)
#
#     # Initialize the dike model from the problem formulation
#     dike_model, planning_steps = get_model_for_problem_formulation(3)
#
#     # Ensure the output directory exists
#     output_dir = os.path.join('data', 'output_data')
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Random Policy Analysis
#     with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
#         random_experiments, random_outcomes = evaluator.perform_experiments(
#             scenarios=100, policies=10
#         )
#     random_experiments.to_csv(os.path.join(output_dir, 'random_experiments.csv'))
#     random_outcomes_df = pd.DataFrame.from_dict(random_outcomes)
#     random_outcomes_df['policy'] = random_experiments['policy']
#     random_outcomes_df.to_csv(os.path.join(output_dir, 'random_outcomes.csv'))
#
#     # Sobol Sensitivity Analysis
#     base_case_policy = [Policy("Do Nothing Policy", **get_do_nothing_dict(dike_model))]
#     with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
#         sobol_experiments, sobol_outcomes = evaluator.perform_experiments(
#             scenarios=256, policies=base_case_policy, uncertainty_sampling=Samplers.SOBOL
#         )
#     sobol_experiments.to_csv(os.path.join(output_dir, 'sobol_experiments.csv'))
#     sobol_outcomes_df = pd.DataFrame.from_dict(sobol_outcomes)
#     sobol_outcomes_df.to_csv(os.path.join(output_dir, 'sobol_outcomes.csv'))
#
#     # No Policy Analysis
#     with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
#         no_policy_experiments, no_policy_outcomes = evaluator.perform_experiments(
#             scenarios=5000, policies=base_case_policy
#         )
#     no_policy_experiments.to_csv(os.path.join(output_dir, 'no_policy_experiments.csv'))
#     no_policy_outcomes_df = pd.DataFrame.from_dict(no_policy_outcomes)
#     no_policy_outcomes_df['policy'] = no_policy_experiments['policy']
#     no_policy_outcomes_df.to_csv(os.path.join(output_dir, 'no_policy_outcomes.csv'))
#