import time
from job_shop_extended.cp_utils import (
    create_job_shop_model,
    solve_job_shop,
    set_scheduled_times_from_job_shop_schedule,
)
from job_shop_extended.instance_parser import (
    get_instance_by_name,
    parse_job_shop_instance,
)
from job_shop_extended.viewer import JobShopViewer
from job_shop_extended.generator import InstanceGenerator
import jax
import matplotlib.pyplot as plt


# Available instance: ft06, la10, orb01, swv01, ta01
instance_name = "ft06"
time_limit = 10_000  # in milliseconds

# Load and prepare instance data
instance_data = get_instance_by_name(instance_name)
num_jobs, num_machines, durations, machine_seq = parse_job_shop_instance(
    instance_data["path"]
)
max_op_duration = max(
    max(op_duration for op_duration in op_durations) for op_durations in durations
)
# MiniZinc uses 1-based indexing so we have to increment the machine ids
model_data = {
    "num_jobs": num_jobs,
    "num_machines": num_machines,
    "num_tasks": num_machines,
    "durations": durations,
    "machine_ids": [[op_id + 1 for op_id in job_op_ids] for job_op_ids in machine_seq],
}

# Setup CP model
model = create_job_shop_model(
    data=model_data,
)

# Find schedule
t0 = time.process_time()
cp_schedule = solve_job_shop(
    model,
    solver_name="gecode",
    solver_config={"--all-solutions": True, "--solver-time-limit": time_limit},
)
print(f"Time to solution: {time.process_time() - t0:.3f}s")
print(cp_schedule._output_item)

# Visualize schedule
instance_state = InstanceGenerator(instance_data["path"])(jax.random.PRNGKey(0))
instance_state_with_schedule = set_scheduled_times_from_job_shop_schedule(
    instance_state, cp_schedule
)
viewer = JobShopViewer(
    f"Schedule for instance {instance_name}",
    num_jobs,
    num_machines,
    num_machines,
    max_op_duration,
)
viewer.render(instance_state)
plt.show()
