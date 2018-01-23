from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import reinforceflow
except ImportError:
    import os.path
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    import reinforceflow
from reinforceflow.agents.async.a3c import A3C
from reinforceflow.envs.wrapper import Vectorize
from reinforceflow.core.policy import EGreedyPolicy
from reinforceflow.models import ActorCriticFC
reinforceflow.set_random_seed(555)

env_name = 'CartPole-v0'
env = Vectorize(env_name)
policies = EGreedyPolicy(eps_start=1.0, eps_final=0.7, anneal_steps=50000)

agent = A3C(env, model=ActorCriticFC(), num_threads=4)

agent.train(log_freq=10,
            test_env=Vectorize(env_name),
            policy=policies,
            maxsteps=50000,
            batch_size=20,
            log_dir='/tmp/rf/A3C/%s' % env_name)

agent.test(Vectorize(env_name), episodes=2, render=True, max_fps=30)
