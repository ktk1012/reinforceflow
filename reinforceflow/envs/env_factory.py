from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from reinforceflow.envs.env_wrapper import EnvWrapper
from reinforceflow.envs.wrappers import AtariWrapper
from reinforceflow import logger


class EnvFactory(object):
    @staticmethod
    def make(env, action_repeat=4, random_start=0):
        if isinstance(env, str):
            env = gym.make(env)
        if hasattr(env.env, 'ale'):
            logger.info('Creating wrapper around Atari environment.')
            return AtariWrapper(stack_len=4,
                                height=84,
                                width=84,
                                action_repeat=action_repeat,
                                to_gray=True,
                                use_merged_frame=True)(env)
        else:
            logger.info('Creating environment wrapper.')
            return EnvWrapper(env)