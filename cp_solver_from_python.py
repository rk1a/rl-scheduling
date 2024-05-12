if __name__ == "__main__":
    import time
    from job_shop_extended.cp_utils import create_job_shop_model, solve_job_shop
    from job_shop_extended.instance_parser import get_instance_by_name, parse_job_shop_instance
    entry = get_instance_by_name("jum01")
    num_jobs, num_machines, durations, machine_seq = parse_job_shop_instance(entry["path"])
    # MiniZinc uses 1-based indexing so we have to increment the machine ids
    model_data = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "num_tasks": num_machines,
        "durations": durations,
        "machine_ids": [[op_id + 1 for op_id in job_op_ids] for job_op_ids in machine_seq],
    }
    model = create_job_shop_model(
        data=model_data,
    )
    t0 = time.process_time()
    cp_schedule = solve_job_shop(model, solver_name="gecode", solver_config={"--all-solutions": True})
    print(f"Time to solution: {time.process_time() - t0:.3f}s")
    print(cp_schedule._output_item)

