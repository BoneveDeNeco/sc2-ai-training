# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-04 15:55:09
# @Last Modified by:   fyr91
# @Last Modified time: 2019-11-24 21:21:24
import gym
from pysc2.env import sc2_env
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features, units
from gym import spaces
from typing import List
import logging
import numpy as np
from pysc2.lib.named_array import NamedNumpyArray

logger = logging.getLogger(__name__)


def get_units_by_type(obs: TimeStep, unit_type, player_relative=features.PlayerRelative.NONE) -> List[NamedNumpyArray]:
    """
    NONE = 0
    SELF = 1
    ALLY = 2
    NEUTRAL = 3
    ENEMY = 4
    """
    return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == player_relative]


class DZBEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "DefeatZerglingsAndBanelings",
        'players': [sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.hard)],
        'agent_interface_format': features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64),
        'realtime': False
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env: sc2_env.SC2Env = None
        self.marines = []
        self.enemies = []
        self.total_reward = 0
        # Action shape indexes:
        # 0: Action
        # 1: Marine index
        # 2: Move delta on x
        # 3: Move delta on y
        # 4: Enemy index
        self.action_space = spaces.Box(
            low=np.array([0, 0, -15, -15, 0]),
            high=np.array([2, 9, 15, 15, 10]),
            dtype=np.int8
        )
        # [0: x, 1: y, 2: hp]
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape=(19, 3),
            dtype=np.uint8
        )

    # Override
    def reset(self):
        if self.env is None:
            self.init_env()

        self.marines = []
        self.enemies = []
        print("Total Reward: ", self.total_reward)
        self.total_reward = 0

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

    def get_derived_obs(self, raw_obs: TimeStep):
        obs = np.zeros((19, 3), dtype=np.uint8)
        marines = get_units_by_type(raw_obs, units.Terran.Marine, features.PlayerRelative.SELF)
        zerglings = get_units_by_type(raw_obs, units.Zerg.Zergling, features.PlayerRelative.ENEMY)
        banelings = get_units_by_type(raw_obs, units.Zerg.Baneling, features.PlayerRelative.ENEMY)
        self.marines = []
        self.enemies = []

        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x, m.y, m[2]])

        for i, b in enumerate(banelings):
            self.enemies.append(b)
            obs[i + 9] = np.array([b.x, b.y, b[2]])

        for i, z in enumerate(zerglings):
            self.enemies.append(z)
            obs[i + 13] = np.array([z.x, z.y, z[2]])

        return obs

    # Override
    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward
        self.total_reward += reward
        obs = self.get_derived_obs(raw_obs)
        # each step will set the dictionary to emtpy
        return obs, reward, raw_obs.last(), {}

    def take_action(self, action):
        action_code = np.int(np.floor(action[0]))
        marine_idx = np.int(np.floor(action[1]))
        if action_code == 0:
            action_mapped = self.move_marine(marine_idx, action[2], action[3])
        else:
            enemy_idx = np.int(np.floor(action[4]))
            action_mapped = self.attack(marine_idx, enemy_idx)

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def move_marine(self, idx, delta_x, delta_y):
        try:
            marine = self.marines[idx]
            new_pos = [marine.x + np.floor(delta_x), marine.y + np.floor(delta_y)]
            return actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, new_pos)
        except Exception as e:
            print("Cannot move: ", e)
            return actions.RAW_FUNCTIONS.no_op()

    def attack(self, aidx, eidx):
        try:
            selected = self.marines[aidx]
            target = self.enemies[eidx]
            return actions.RAW_FUNCTIONS.Attack_unit("now", selected.tag, target.tag)
        except Exception as e:
            print("Cannot attack: ", e)
            return actions.RAW_FUNCTIONS.no_op()

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    # Override
    def render(self, mode='human', close=False):
        pass
