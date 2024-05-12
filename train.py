# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
from job_shop_extended.cp_utils import (
    create_job_shop_model,
    solve_job_shop,
    set_scheduled_times_from_job_shop_schedule,
    state_to_cp_model_data,
)
from job_shop_extended.instance_parser import get_instance_by_name
import functools
import logging
from typing import Dict, Tuple

import optax
import hydra
import jax
import jax.numpy as jnp
import omegaconf
from tqdm.auto import trange

from jumanji.training import utils
from jumanji.training.agents.base import Agent
from job_shop_extended.cp_utils import state_to_cp_model_data
from job_shop_extended.evaluator import Evaluator
from jumanji.training.setup_train import (
    setup_agent,
    setup_logger,
    setup_training_state,
)
from jumanji.training.timer import Timer
from jumanji.training.types import TrainingState

from typing import Tuple

import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from job_shop_extended.env import JobShop
from job_shop_extended.generator import (
    RandomGenerator,
    InstanceGenerator,
    RandomDurationInstanceGenerator,
)
from jumanji.training.agents.random import RandomAgent
from jumanji.training.loggers import TerminalLogger, TensorboardLogger
from jumanji.training.types import TrainingState
from jumanji.wrappers import VmapAutoResetWrapper
from jumanji.training.agents.a2c import A2CAgent
from job_shop_extended.job_shop_network import make_actor_critic_networks_job_shop


def _make_raw_env(cfg: DictConfig) -> JobShop:
    if cfg.env.instance.name is not None:
        instance_data = get_instance_by_name(cfg.env.instance.name)
        if cfg.env.instance.random_duration_change == 0.0:
            generator = InstanceGenerator(instance_data["path"])
        else:
            generator = RandomDurationInstanceGenerator(
                instance_data["path"], cfg.env.instance.random_duration_change
            )
    else:
        generator = RandomGenerator()
    return JobShop(
        generator=generator,
        slowdown_probability=cfg.env.slowdown.prob_per_step,
        slowdown_factor=cfg.env.slowdown.percentage,
    )


def setup_env(cfg: DictConfig) -> JobShop:
    env = _make_raw_env(cfg)
    env = VmapAutoResetWrapper(env)
    return env


def setup_evaluators(cfg: DictConfig, agent: Agent) -> Tuple[Evaluator, Evaluator]:
    env = _make_raw_env(cfg)
    stochastic_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.eval_total_batch_size,
        stochastic=True,
    )
    greedy_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.greedy_eval_total_batch_size,
        stochastic=False,
    )
    return stochastic_eval, greedy_eval


def setup_agent(cfg: DictConfig, env: JobShop) -> Agent:
    agent: Agent
    actor_critic_networks = make_actor_critic_networks_job_shop(
        job_shop=env.unwrapped,
        num_layers_machines=cfg.env.network.num_layers_machines,
        num_layers_operations=cfg.env.network.num_layers_operations,
        num_layers_joint_machines_jobs=cfg.env.network.num_layers_joint_machines_jobs,
        transformer_num_heads=cfg.env.network.transformer_num_heads,
        transformer_key_size=cfg.env.network.transformer_key_size,
        transformer_mlp_units=cfg.env.network.transformer_mlp_units,
    )
    optimizer = optax.adam(cfg.env.a2c.learning_rate)
    agent = A2CAgent(
        env=env,
        n_steps=cfg.env.training.n_steps,
        total_batch_size=cfg.env.training.total_batch_size,
        actor_critic_networks=actor_critic_networks,
        optimizer=optimizer,
        normalize_advantage=cfg.env.a2c.normalize_advantage,
        discount_factor=cfg.env.a2c.discount_factor,
        bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
        l_pg=cfg.env.a2c.l_pg,
        l_td=cfg.env.a2c.l_td,
        l_en=cfg.env.a2c.l_en,
    )
    return agent


@hydra.main(config_path="configs", config_name="config.yaml")
def train(cfg: omegaconf.DictConfig, log_compiles: bool = False) -> None:
    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    logging.info({"devices": jax.local_devices()})

    key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    logger = setup_logger(cfg)
    env = setup_env(cfg)
    agent = setup_agent(cfg, env)
    stochastic_eval, greedy_eval = setup_evaluators(cfg, agent)
    training_state = setup_training_state(env, agent, init_key)
    num_steps_per_epoch = (
        cfg.env.training.n_steps
        * cfg.env.training.total_batch_size
        * cfg.env.training.num_learner_steps_per_epoch
    )
    eval_timer = Timer(out_var_name="metrics")
    train_timer = Timer(
        out_var_name="metrics", num_steps_per_timing=num_steps_per_epoch
    )

    @functools.partial(jax.pmap, axis_name="devices")
    def epoch_fn(training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        training_state, metrics = jax.lax.scan(
            lambda training_state, _: agent.run_epoch(training_state),
            training_state,
            None,
            cfg.env.training.num_learner_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, metrics

    optimum_makespan: int | None = None
    if isinstance(logger, TensorboardLogger):
        layout = {
            "Baseline": {
                "makespan": ["Multiline", ["a2c/makespan", "cp/makespan", "optimum/makespan"]],
            },
        }
        logger.writer.add_custom_scalars(layout)
        if cfg.env.instance.name is not None:
            instance_data = get_instance_by_name(cfg.env.instance.name)
            optimum_makespan = instance_data["optimum"]

    with jax.log_compiles(log_compiles), logger:
        for i in trange(
            cfg.env.training.num_epochs,
            disable=isinstance(logger, TerminalLogger),
        ):
            env_steps = i * num_steps_per_epoch

            # Evaluation
            key, stochastic_eval_key, greedy_eval_key = jax.random.split(key, 3)
            # Stochastic evaluation
            with eval_timer:
                metrics = stochastic_eval.run_evaluation(
                    training_state.params_state, stochastic_eval_key
                )
                jax.block_until_ready(metrics)

            logger.write(
                data=utils.first_from_device(metrics),
                label="eval_stochastic",
                env_steps=env_steps,
            )
            if not isinstance(agent, RandomAgent):
                # Greedy evaluation
                with eval_timer:
                    metrics = greedy_eval.run_evaluation(
                        training_state.params_state, greedy_eval_key
                    )
                    jax.block_until_ready(metrics)
                logger.write(
                    data=utils.first_from_device(metrics),
                    label="eval_greedy",
                    env_steps=env_steps,
                )

            # Log schedule images and makespans of evaluation instance with lowest makespan
            if isinstance(logger, TensorboardLogger):
                best_final_states = metrics["best_final_state"]
                for eval_i in range(best_final_states.step_count.shape[0]):
                    best_final_state = jax.tree_util.tree_map(
                        lambda x: x.at[i].get(), best_final_states
                    )
                    a2c_makespan = best_final_state.step_count
                    fig_a2c_schedule = env.unwrapped._viewer.render(best_final_state)
                    logger.writer.add_figure(
                        f"a2c/best_schedule_epoch{i}_eval{eval_i}_makespan{a2c_makespan}",
                        fig_a2c_schedule,
                        global_step=env_steps,
                    )
                    logger.writer.add_scalar("a2c/makespan", a2c_makespan, env_steps)
                    if optimum_makespan is not None:
                        logger.writer.add_scalar("optimum/makespan", optimum_makespan, env_steps)

                    if cfg.env.evaluation.cp_baseline:
                        # Note: we assume that the input data didn't change during the rollout
                        model_data = state_to_cp_model_data(best_final_state)
                        model = create_job_shop_model(
                            data=model_data,
                        )
                        # Solve the job shop instance with CP for the given time
                        # Note: for very large instances the solver might not find a solution
                        # within the given time limit
                        cp_schedule = solve_job_shop(
                            model,
                            solver_name=cfg.env.evaluation.cp_solver,
                            solver_config={
                                "--solver-time-limit": cfg.env.evaluation.cp_timelimit
                            },
                        )
                        cp_makespan = cp_schedule.objective
                        state_with_cp_schedule = set_scheduled_times_from_job_shop_schedule(
                            best_final_state, cp_schedule
                        )
                        fig_cp_schedule = env.unwrapped._viewer.render(
                            state_with_cp_schedule
                        )
                        logger.writer.add_figure(
                            f"cp/schedule_epoch{i}_eval{eval_i}_makespan{cp_makespan}",
                            fig_cp_schedule,
                            global_step=env_steps,
                        )
                        logger.writer.add_scalar("cp/makespan", cp_makespan, env_steps)


            # Training
            with train_timer:
                training_state, metrics = epoch_fn(training_state)
                jax.block_until_ready((training_state, metrics))
            logger.write(
                data=utils.first_from_device(metrics),
                label="train",
                env_steps=env_steps,
            )


if __name__ == "__main__":
    train()
