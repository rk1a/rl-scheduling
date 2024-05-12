from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from minizinc import Instance, Model, Solver, Status
import pathlib

from job_shop_extended.types import State


@dataclass
class JobShopSchedule:
    start_times: list[list[int]]  # noqa: N815
    end: int
    objective: int
    _output_item: str


def create_job_shop_model(
    model_file: pathlib.Path | str = pathlib.Path("minizinc/job_shop.mzn").absolute(),
    data_files: list[pathlib.Path | str] | None = None,
    data: Mapping[str, Any] | None = None,
    output_type: type = JobShopSchedule,
    enum_types: list[type] | None = None,
) -> Model:
    model = Model([model_file])
    if output_type:
        model.output_type = output_type
    if enum_types:
        for enum_type in enum_types:
            model[enum_type.__name__] = enum_type
    if data_files:
        for data_file in data_files:
            model.add_file(data_file, parse_data=True)
    if data:
        for parameter, value in data.items():
            model[parameter] = value
    return model

def solve_job_shop(
    model: Model,
    solver_name: str = "gecode",
    solver_config: Mapping[str, Any] | None = None,
) -> JobShopSchedule:
    """Use solver for finding a schedule to job shop instance.

    Args:
        model: Model with loaded job shop instance data.
        solver_name: Name of the solver.
        solver_config: Configuration of the solver.

    Returns:
        JobShopSchedule object.
    
    Raises:
        ValueError if no solution was found.
    """
    solver_config = solver_config or {}
    solver = Solver.lookup(solver_name)
    instance = Instance(solver, model)

    result = instance.solve(**solver_config)

    if (
        result.status in [Status.OPTIMAL_SOLUTION, Status.SATISFIED]
        and result.solution is not None
    ):
        return result.solution
    else:
        raise ValueError(f"No solution found with status {result.status}.")


def state_to_cp_model_data(state: State) -> dict[str, any]:
    """Converts environment state into MiniZinc compatible model data."""
    num_jobs, num_tasks = state.ops_durations.shape
    num_machines = state.machines_job_ids.shape[0]
    durations = jnp.where(state.ops_durations == -1, 0, state.ops_durations).tolist()
    machine_ids = (state.ops_machine_ids + 1).tolist()
    return {
        "num_jobs": num_jobs,
        "num_tasks": num_tasks,
        "num_machines": num_machines,
        "durations": durations,
        "machine_ids": machine_ids,
    }

def set_scheduled_times_from_job_shop_schedule(state: State, job_shop_schedule: JobShopSchedule) -> State:
    """Set schedule attributes in state from schedule obtained from CP solver."""
    job_op_start_times = job_shop_schedule.start_times
    for job_id, op_start_times in enumerate(job_op_start_times):
        for op_id, op_start_time in enumerate(op_start_times):
            state.scheduled_times = state.scheduled_times.at[job_id, op_id].set(op_start_time)
    state.step_count = job_shop_schedule.objective
    return state