import json
import pathlib
from hydra.utils import to_absolute_path

def get_instance_by_name(instance_name: str, instance_data_file: str = to_absolute_path("instances/instances.json")):
    """
    Retrieve instance data of the given instance name.

    Args:
        name: The name of the entry to be retrieved.
        instance_data_file: Path to the instance data JSON file.

    Returns:
        A dictionary containing the entry data if found, otherwise None.
    """
    with open(instance_data_file) as json_file:
        data = json.load(json_file)
    for entry in data:
        if entry["name"] == instance_name:
            return entry
    return None

def parse_job_shop_instance(
    file_path: str,
) -> tuple[int, int, list[list[int]], list[list[int]]]:
    """
    Parses a job shop instance file.

    Args:
        file_path: The path to the job shop instance file.

    Returns:
        A tuple containing:
            - num_jobs: The number of jobs in the instance (int).
            - num_machines: The number of machines in the instance (int).
            - processing_times: A list of lists of integers, where processing_times[i]
                               represents the processing times for each task of job i.
            - machine_sequence: A list of lists of integers, where machine_sequence[i]
                               represents the sequence of machines for each task of job i.
    """
    abs_file_path = to_absolute_path(file_path)
    abs_file_path = pathlib.Path(abs_file_path)
    if not abs_file_path.exists():
        raise ValueError(f"Couldn't find instance at {abs_file_path}!")
    with open(abs_file_path, "r") as f:
        # Skip comments
        line = ""
        while True:
            line = f.readline()
            if not line.startswith('#'):
                break
        # Read the first line containing number of jobs and machines
        num_jobs, num_machines = map(int, line.split())

        # Initialize empty lists to store processing times and machine sequences
        processing_times = []
        machine_sequence = []

        # Read each line for a job
        for line in f:
            job_data = list(map(int, line.split()))

            # Split job data into processing times and machine sequence
            processing_times.append(job_data[1::2])
            machine_sequence.append(job_data[::2])

    return num_jobs, num_machines, processing_times, machine_sequence
