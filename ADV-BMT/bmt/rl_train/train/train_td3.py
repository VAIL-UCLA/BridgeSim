import argparse
import os
import random
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import wandb
from IPython.display import clear_output
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.scenario.utils import get_number_of_scenarios
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import ResultsWriter
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import sync_envs_normalization
from wandb.integration.sb3 import WandbCallback

from bmt.rl_train.train.ScenarioOnlineEnvWrapper import ScenarioOnlineEnvWrapper
from bmt.rl_train.train.customized_td3 import CustomizedTD3, Closed_Loop_TD3


# from metadrive.engine.logger import set_log_level
# set_log_level(logging.ERROR)

def set_seed(seed):
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # PyTorch random (CPU and GPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch random (all GPU devices)
    torch.backends.cudnn.deterministic = True  # Use deterministic CuDNN operations
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmark for determinism


class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
            self,
            env,
            filename: Optional[str] = None,
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            # info_keywords: Tuple[str, ...] = (),
    ):
        super(Monitor, self).__init__(env=env)

        # PZH: Step the environment for once to understand the info keys.
        self.env.reset()
        o, r, tm, tc, i = self.env.step(self.env.action_space.sample())
        info_keywords = tuple(i.keys())
        reset_keywords = tuple(reset_keywords)
        ep_info_keywords = tuple("ep_" + k for k in info_keywords)
        record_keys = reset_keywords + info_keywords + ep_info_keywords

        self.t_start = time.time()
        if filename is not None:
            self.results_writer = ResultsWriter(
                filename,
                header={
                    "t_start": self.t_start,
                    "env_id": env.spec and env.spec.id
                },
                extra_keys=record_keys,
            )
        else:
            self.results_writer = None
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.metadata["info_keywords"] = self.info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_times = []

        # PZH: Ours
        self.episode_infos = defaultdict(list)

        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, tm, tc, info = self.env.step(action)
        self.rewards.append(reward)

        for key in self.info_keywords:
            self.episode_infos[key].append(info[key])

        done = tm or tc
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
                ep_data = np.asarray(self.episode_infos[key])
                if ep_data.dtype == object:
                    pass
                else:
                    # Temporary workaround solution for accessing mean for non float/int
                    try:
                        ep_info["epavg_{}".format(key)] = np.mean(ep_data)
                    except TypeError:
                        pass
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
            self.episode_infos.clear()
        self.total_steps += 1
        return observation, reward, tm, tc, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super(Monitor, self).close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times


def evaluate_info(all_info):
    all_scenarios = list(all_info.keys())
    num_scenario = len(all_scenarios)

    _rewards = []
    _costs = []
    _completion = []
    _crash = []
    _episode_length = []

    for i in all_scenarios:
        _rewards.append(all_info[i]['episode_reward'])
        _costs.append(all_info[i]['cost'])
        _completion.append(all_info[i]['route_completion'])
        _crash.append(1 if all_info[i]['crash'] else 0)
        _episode_length.append(all_info[i]['episode_length'])

    # Convert lists to NumPy arrays
    _rewards = np.array(_rewards)
    _costs = np.array(_costs)
    _completion = np.array(_completion)
    _crash = np.array(_crash)
    _episode_length = np.array(_episode_length)

    result = {
        "num_scenario": num_scenario,
        "avg_rewards": np.mean(_rewards),
        "avg_costs": np.mean(_costs),
        "avg_completion": np.mean(_completion),
        "avg_collisions": np.mean(_crash),
        "avg_length": np.mean(_episode_length),
    }

    return result


class CustomizedFormalevalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluations_info_buffer = defaultdict(list)
        self._all_episode_info = {}
        self._episode_counter = 0

    def _log_success_callback(self, locals_, globals_):
        info = locals_["info"]
        if locals_["done"]:
            info = dict(info)  # Shallow copy

            completion = info['route_completion']
            if completion <= 0:
                info['route_completion'] = 0

            if completion >= 1:
                info['route_completion'] = 1

            # if info['arrive_dest']:
            #     info['route_completion'] = 1

            info['crash'] = bool(info.get("crash", False))

            self._all_episode_info[self._episode_counter] = info
            self._episode_counter += 1

            for k in ["route_completion", "cost", "arrive_dest", "out_of_road", "crash", "episode_reward"]:
                if k in info:
                    self.evaluations_info_buffer[k].append(info[k])

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync normalization if needed
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval envs must be wrapped similarly for normalization."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            self.evaluations_info_buffer.clear()

            print("Start evaluating policy for {} episodes!".format(self.n_eval_episodes))

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            print("Finish evaluating policy for {} episodes!".format(self.n_eval_episodes))

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs["successes"] = self.evaluations_successes

                for k, v in self.evaluations_info_buffer.items():
                    assert len(v) <= self.n_eval_episodes
                    kwargs[k] = v

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)
            std_ep_length = np.std(episode_lengths)

            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # self.logger.record("eval/mean_reward", float(mean_reward))
            # self.logger.record("eval/mean_ep_length", mean_ep_length)
            # self.logger.record("eval/num_episodes", len(episode_rewards))

            # if len(self._is_success_buffer) > 0:
            #     success_rate = np.mean(self._is_success_buffer)
            #     if self.verbose >= 1:
            #         print(f"Success rate: {100 * success_rate:.2f}%")
            #     self.logger.record("eval/success_rate", success_rate)

            results_dict = evaluate_info(self._all_episode_info)
            self._all_episode_info = {}  # Reset
            self._episode_counter = 0

            for k, v in results_dict.items():
                self.logger.record(f"eval/{k}", v)

            for k, v in self.evaluations_info_buffer.items():
                assert len(v) == self.n_eval_episodes
                self.logger.record(f"eval/{k}", np.mean(np.asarray(v)))

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


def create_env(config, need_monitor=False, closed_loop=False, closed_loop_generator="SCGEN", model_name=None,
               no_adaptive=False):
    if closed_loop:
        if closed_loop_generator == "SCGEN":
            from bmt.rl_train.train.scgen_generator import SCGEN_Generator
            generator = SCGEN_Generator()

        elif closed_loop_generator == "CAT":
            from bmt.rl_train.train.cat_generator import CAT_Generator
            generator = CAT_Generator()
        else:
            raise ValueError("Unknown closed_loop_generator")

        env = ScenarioOnlineEnvWrapper(config=config, generator=generator, no_adaptive=no_adaptive)
    else:
        env = ScenarioEnv(config=config)

    if need_monitor:
        env = Monitor(env)  # Pass the custom metrics

    return env


def create_eval_env(eval_config, need_monitor=False, ):
    eval_env = ScenarioEnv(config=eval_config)

    if need_monitor:
        eval_env = Monitor(eval_env)  # Pass the custom metrics

    return eval_env


class WandbLoggingCallback(BaseCallback):
    """Logs TD3 loss and other key training metrics to Weights & Biases (W&B)."""

    def __init__(self, verbose=1):
        super(WandbLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """Logs training metrics every step."""
        if "loss/critic" in self.locals:
            wandb.log({"loss/critic": self.locals["loss/critic"]})
        if "loss/policy" in self.locals:
            wandb.log({"loss/policy": self.locals["loss/policy"]})

        return True


def train(
        config_train, config_eval, load_model_path=None, seed=None, save_path="./td3", training_steps=None,
        lr=None, eval_freq=None, eval_ep=None, wandb_config=None, exp_name="td3", num_eval_envs=None):
    assert seed is not None
    assert num_eval_envs is not None
    set_seed(seed)
    train_env = DummyVecEnv([lambda: create_env(config_train, True)])  # use only one training environment
    save_prefix = f"seed_{seed}"
    callbacks = []
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=save_path, name_prefix=save_prefix)
    use_wandb = wandb_config.get("use_wandb", False)
    if use_wandb:
        import wandb
        project_name = wandb_config.get("wandb_project", "scgen")
        team_name = wandb_config.get("wandb_team", "drivingforce")
        wandb.init(
            project=project_name,
            entity=team_name,
            name=f"{exp_name}_seed_{seed}",
            group=exp_name,
            sync_tensorboard=True,
            save_code=True
        )
        wandb_callback = WandbCallback(model_save_path=f"./wandb_models/{exp_name}_seed_{seed}", verbose=1)
        wandb_loss_callback = WandbLoggingCallback()
        callbacks.append(wandb_callback)
        callbacks.append(wandb_loss_callback)

    eval_env = SubprocVecEnv([lambda: create_eval_env(config_eval) for _ in range(num_eval_envs)])

    eval_callback = CustomizedFormalevalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=eval_ep,
        best_model_save_path=save_path,
        log_path=save_path,
        # deterministic=True,
        # render=False
    )

    callbacks.append(checkpoint_callback)
    callbacks.append(eval_callback)

    if load_model_path:
        model = CustomizedTD3.load(load_model_path, env=train_env)
        print(f"Resuming training from model at {load_model_path}")

        trained_steps = int(model.num_timesteps)

        remaining_steps = training_steps - trained_steps
        print(f"Resuming from {trained_steps} steps; training for {remaining_steps} more steps.")

        model.learn(
            total_timesteps=remaining_steps,
            reset_num_timesteps=False,  # Important: continue counting from previous steps
            callback=callbacks,
        )

    else:
        model = CustomizedTD3("MlpPolicy",
                              train_env,
                              action_noise=None,
                              learning_rate=lr,
                              learning_starts=200,
                              batch_size=1024,
                              tau=0.005,
                              gamma=0.99,
                              train_freq=1,
                              gradient_steps=1,
                              device="cuda",
                              seed=seed,
                              verbose=2,
                              tensorboard_log="TD3",
                              )
        print("Starting new training...")

    model.learn(
        total_timesteps=training_steps,
        callback=callbacks,
    )

    clear_output()


def closed_loop_train(
        config_train, config_eval, load_model_path=None, seed=None, save_path=None,
        training_steps=None, lr=None, eval_freq=None, eval_ep=None, wandb_config=None,
        exp_name="td3", source_data=None, closed_loop_generator="SCGEN",
        model_name=None, resumed_step=0, num_eval_envs=None, no_adaptive=False):
    assert seed is not None
    assert eval_ep is not None
    set_seed(seed)

    train_env = create_env(config_train, need_monitor=True, closed_loop=True,
                           closed_loop_generator=closed_loop_generator, model_name=model_name, no_adaptive=no_adaptive)
    save_prefix = f"seed_{seed}"

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=save_path, name_prefix=save_prefix)

    callbacks = [checkpoint_callback]

    use_wandb = wandb_config.get("use_wandb", False)
    if use_wandb:
        project_name = wandb_config.get("wandb_project", "scgen")
        team_name = wandb_config.get("wandb_team", "drivingforce")

        if load_model_path is not None and resumed_step > 0:
            resumed_step = resumed_step
            wandb.init(
                # id=resume_id,
                # resume="must",
                project="scgen",
                entity="drivingforce",
                name=f"{exp_name}_seed_{seed}_resumed_{resumed_step}",
                group=exp_name,
                sync_tensorboard=True,
                save_code=True  # Save script files in W&B
            )
        else:
            wandb.init(
                project=project_name,
                entity=team_name,
                name=f"{exp_name}_seed_{seed}",
                group=exp_name,
                sync_tensorboard=True,
                save_code=True  # Save script files in W&B
            )

        wandb_callback = WandbCallback(model_save_path=f"./wandb_models/{exp_name}_seed_{seed}", verbose=1)
        callbacks.append(wandb_callback)

    # eval_env = None
    eval_env = SubprocVecEnv([(lambda: create_eval_env(config_eval)) for _ in range(num_eval_envs)])

    if eval_env is not None:
        eval_callback = CustomizedFormalevalCallback(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=eval_ep,
            best_model_save_path=save_path,
            log_path=save_path,
            # deterministic=True,
            # render=False
        )
        callbacks.append(eval_callback)

    callbacks.append(checkpoint_callback)

    if load_model_path:
        model = Closed_Loop_TD3.load(load_model_path, env=train_env)
        print(f"Resuming training from model at {load_model_path}")

        trained_steps = int(model.num_timesteps)
        remaining_steps = training_steps - trained_steps
        print(f"Resuming from {trained_steps} steps; training for {remaining_steps} more steps.")

        model.learn(
            total_timesteps=remaining_steps,
            reset_num_timesteps=False,  # Important: continue counting from previous steps
            callback=callbacks,
        )

    else:
        model = Closed_Loop_TD3("MlpPolicy",
                                train_env,
                                action_noise=None,
                                learning_rate=lr,
                                learning_starts=200,
                                batch_size=1024,
                                tau=0.005,
                                gamma=0.99,
                                train_freq=1,
                                gradient_steps=1,
                                device="cuda",
                                seed=seed,
                                verbose=2,
                                tensorboard_log=str(save_path),
                                training_dataset=source_data
                                )
        print("Starting new training...")

        model.learn(
            total_timesteps=training_steps,
            callback=callbacks,
        )

    clear_output()


def train_wrapper(
        config_train, config_eval, exp_name, seed, save_path, ckpt_path=None,
        training_steps=None, eval_freq=None, lr=None, wandb_config=None, closed_loop=False,
        closed_loop_dir=None, closed_loop_generator="SCGEN", model_name=None, resumed_step=0,
        num_eval_envs=None, eval_ep=None, no_adaptive=False):
    print("current learning rate:", lr)

    if not closed_loop:
        train(
            config_train=config_train,
            config_eval=config_eval,
            load_model_path=ckpt_path,
            seed=seed,
            save_path=save_path,
            training_steps=training_steps,
            lr=lr,
            eval_freq=eval_freq,
            wandb_config=wandb_config,
            exp_name=exp_name,
            num_eval_envs=num_eval_envs,
            eval_ep=eval_ep,
        )

    else:
        assert closed_loop_dir is not None, "Please provide the closed loop source data directory."
        closed_loop_train(
            config_train=config_train,
            config_eval=config_eval,
            load_model_path=ckpt_path,
            seed=seed,
            save_path=save_path,
            training_steps=training_steps,
            lr=lr,
            eval_freq=eval_freq,
            wandb_config=wandb_config,
            exp_name=exp_name,
            source_data=closed_loop_dir,
            closed_loop_generator=closed_loop_generator,
            model_name=model_name,
            resumed_step=resumed_step,
            num_eval_envs=num_eval_envs,
            eval_ep=eval_ep,
            no_adaptive=no_adaptive
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The dir for this batch of experiments.")
    parser.add_argument("--eval_data_dir", type=str, help="The dir for this batch of experiments.")
    parser.add_argument("--save_path", type=str, help="The dir for checkpoints to save.")
    parser.add_argument("--exp_name", default="td3_metadrive", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--training_step", default=1_000_000, type=int, help="The number of steps in training")
    parser.add_argument("--eval_freq", default=50000, type=int, help="Eval frequency.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of TD3.")
    parser.add_argument("--CAT_config", action="store_true", help="Set to CAT train config")
    parser.add_argument("--horizon", type=int, help="training horizon")
    parser.add_argument("--eval_horizon", type=int, default=100, help="training horizon")
    parser.add_argument("--ckpt_path", type=str, help="pre-trained policy path")
    parser.add_argument("--closed_loop", action="store_true", help="closd_loop")
    parser.add_argument("--source_data", type=str, help="closd_loop source data directory")
    parser.add_argument("--closed_loop_generator", type=str, help="closed_loop_generator")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--resume_wandb_id", type=str, help="wandb run id to resume training")
    parser.add_argument("--resumed_step", type=int, help="resuming step number.")
    parser.add_argument("--num_eval_envs", type=int, help="resuming step number.")
    parser.add_argument("--eval_ep", type=int, help="number of eval episodes.")
    parser.add_argument("--no_adaptive", action="store_true", help="generator takes GT ego traj")

    args = parser.parse_args()

    num_scenario = get_number_of_scenarios(args.data_dir)
    print(f"Number of scenarios: {num_scenario}")
    print(f"Number of training horizon: {args.horizon}")

    # Assert gpu is there
    assert torch.cuda.is_available(), "GPU is not available. Please check your CUDA installation."

    config_train = dict(
        store_map=False,
        use_render=False,
        manual_control=False,
        show_interface=False,
        data_directory=args.data_dir,
        agent_policy=EnvInputPolicy,
        start_scenario_index=0,
        num_scenarios=num_scenario,
        sequential_seed=False,
        horizon=args.horizon,
        reactive_traffic=False,
        no_static_vehicles=True,
        no_light=True,
        # crash_vehicle_done=True,
        # out_of_route_done=True,
        # crash_object_done=True,
        # crash_human_done=False,
        # relax_out_of_road_done=False,
    )

    config_eval = dict(
        store_map=False,
        use_render=False,
        manual_control=False,
        show_interface=False,
        data_directory=args.eval_data_dir,
        agent_policy=EnvInputPolicy,
        start_scenario_index=0,
        num_scenarios=get_number_of_scenarios(args.eval_data_dir),
        sequential_seed=True,
        horizon=args.eval_horizon,
        reactive_traffic=False,
        no_static_vehicles=True,
        no_light=True,
        crash_vehicle_done=False,
        out_of_route_done=False,
        crash_object_done=False,
        crash_human_done=False,
        relax_out_of_road_done=False,
    )

    if args.closed_loop:
        config_train["total_timesteps"] = args.training_step

    wandb_config = {
        "use_wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_team": args.wandb_team,
    }

    train_wrapper(
        config_train=config_train,
        config_eval=config_eval,
        exp_name=args.exp_name,
        seed=args.seed,
        save_path=args.save_path,
        ckpt_path=args.ckpt_path,
        training_steps=args.training_step,
        lr=args.lr,
        eval_freq=args.eval_freq,
        wandb_config=wandb_config,
        closed_loop=args.closed_loop,
        closed_loop_dir=args.source_data,
        closed_loop_generator=args.closed_loop_generator,
        model_name=args.model_name,
        resumed_step=args.resumed_step,
        num_eval_envs=args.num_eval_envs,  # TODO: num_eval_envs just set it to 5?
        eval_ep=args.eval_ep,  # TODO: this is wrong. it should be eval_horizon, not eval_ep.
        no_adaptive=args.no_adaptive
    )
