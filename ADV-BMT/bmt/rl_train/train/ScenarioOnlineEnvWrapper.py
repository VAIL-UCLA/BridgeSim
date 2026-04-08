from metadrive.envs.scenario_env import ScenarioOnlineEnv
from typing import Union, Dict, AnyStr
from metadrive.scenario.utils import get_number_of_scenarios
import os
import pickle
from metadrive.scenario.utils import read_dataset_summary, read_scenario_data
import numpy as np
import random
import copy


def get_filenames(folder_path, prefix="sd_"):
    all_files = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)) and file.startswith(prefix):
            all_files.append(os.path.join(folder_path, file))
                          
    return all_files

class ScenarioOnlineEnvWrapper(ScenarioOnlineEnv):
    def default_config(cls):
        config = super().default_config()
        config.update(
            {
                "total_timesteps": 2_000_000,
                "min_prob": 0.5,
                "store_map": False
            }
        )
        return config

    def __init__(self, generator=None, config=None, no_adaptive=False):
        self.scenario_dataset = config["data_directory"]
        super(ScenarioOnlineEnvWrapper, self).__init__(config)
        if no_adaptive:
            generator.set_no_adaptive(True)

        self.generator = generator
        self.scenario_index = 0
        # print("num of scenarios:", self.num_scenarios)
        self.all_scenario_files = get_filenames(self.scenario_dataset)
        random.shuffle(self.all_scenario_files)

        self._total_timesteps = self.config["total_timesteps"]
        self.num_timesteps = 0
        self.min_prob = self.config["min_prob"]
        self.in_raw_scenario = False

        assert self.config["store_map"] is False, "store_map should be False in ScenarioOnlineEnvWrapper"
        

    def set_total_time_steps(self, total_steps):
        raise ValueError("please don't use this function, set config please...")
        self._total_timesteps = total_steps


    def set_timestep(self, step):
        self.num_timesteps = step

    def reset(self, seed: Union[None, int] = None):

        if self.generator.ego_traj:  # first time reset() does not need after_episode
            self.generator.after_episode() # update current ego traj for current scenario

        original_SD_path = self.all_scenario_files[self.scenario_index % self.num_scenarios]
        scenario_description = read_scenario_data(original_SD_path)

        self.set_scenario(scenario_description) # by default use the origianl SD

        self.generator.before_episode(self) # parse GT info if not done before

        assert self.engine.data_manager.current_scenario['id'] == scenario_description['id'], "Scenario ID mismatch"

        progress = max(min(self.num_timesteps / self._total_timesteps, 1), 0)
        prob = self.min_prob * progress  # (0 -> min_prob)

        assert len(
            self.all_scenario_files) == self.num_scenarios, "The number of scenarios is not equal to the number of scenario files"

        if np.random.random() < prob:
            print(
                "Current step: {}, Total steps: {}, Progress: {:.2f}, Probability to generate: {:.2f}/{}. Current scenario index {}/{}. Generating...".format(
                    self.num_timesteps, self._total_timesteps, progress, prob, self.min_prob, self.scenario_index,
                    self.num_scenarios
                ))
            new_SD = self.generator.generate()
            if new_SD is None:
                pass
            else:
                self.set_scenario(new_SD)
            self.in_raw_scenario = False
        else:
            print(
                "Current step: {}, Total steps: {}, Progress: {:.2f}, Probability to generate: {:.2f}/{}. Current scenario index {}/{}. Use original scenario.".format(
                    self.num_timesteps, self._total_timesteps, progress, prob, self.min_prob, self.scenario_index,
                    self.num_scenarios
                ))
            self.in_raw_scenario = True

        self.scenario_index += 1
        o, i = super().reset()
        i["in_raw_scenario"] = self.in_raw_scenario
        return o, i

    def step(self, *args, **kwargs):
        ret = super().step(*args, **kwargs)
        ret[-1]["in_raw_scenario"] = self.in_raw_scenario
        return ret
