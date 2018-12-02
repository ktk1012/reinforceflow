from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
from operator import itemgetter
import random
import numpy as np
from reinforceflow.core.datastruct import SumTree, MinTree
from reinforceflow import logger


class ExperienceReplay(object):
    """Experience replay buffer.

    Args:
        capacity (int):  Total replay capacity.
        batch_size (int): Size of sampled batch.
        min_size (int): Minimum replay size (enables is_ready property, when fills).
    """
    def __init__(self, capacity, batch_size, min_size=0):
        if batch_size < 1:
            raise ValueError("Batch size must be higher or equal to 1.")
        if capacity < batch_size:
            logger.warn("Minimum capacity must be higher or equal "
                        "to the batch size (Got: %s). "
                        "Setting minimum buffer size to the batch size." % capacity)
            capacity = batch_size
        self._capacity = capacity
        self._batch_size = batch_size
        self.min_size = max(batch_size, min_size)
        # Python lists offers ~18% faster index access speed at current setup,
        # at the same time sacrificing ~18% of memory compared to numpy.ndarray.
        self._obs = [0] * (capacity + 1)
        self._actions = [0] * capacity
        self._rewards = [0] * capacity
        self._terms = [0] * capacity
        self._idx = 0
        self._size = 0

    def _cycle_idx(self, idx):
        return idx % self._capacity

    def add(self, obs, action, reward, term, obs_next):
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._terms[self._idx] = term
        self._obs[self._idx] = obs
        self._obs[self._idx + 1] = obs if term else obs_next
        self._idx = self._cycle_idx(self._idx + 1)
        self._size = min(self._size + 1, self._capacity)

    def sample(self):
        rand_idxs = random.sample(range(self._size), self._batch_size)
        gather = itemgetter(*rand_idxs)
        next_obs_gather = itemgetter(*[i + 1 for i in rand_idxs])
        return (gather(self._obs),
                gather(self._actions),
                gather(self._rewards),
                gather(self._terms),
                next_obs_gather(self._obs),
                np.ones_like(rand_idxs, 'bool'),
                rand_idxs,
                [1.0] * len(rand_idxs))
        # traj = TrajectoryBatch(obses=gather(self._obs),
        #                        actions=gather(self._actions),
        #                        rewards=gather(self._rewards),
        #                        terms=gather(self._terms),
        #                        next_obses=next_obs_gather(self._obs),
        #                        ends=np.ones_like(rand_idxs, 'bool'))
        # return traj

    @property
    def size(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def is_ready(self):
        return self._size >= self.min_size

    def __len__(self):
        return self._size


class ProportionalReplay(ExperienceReplay):
    """Proportional Prioritized Experience replay buffer.
    Based on paper: https://arxiv.org/pdf/1511.05952.pdf
    Args:
        capacity (int):  Total replay capacity.
        batch_size (int): Size of sampled batch.
        min_size (int): Minimum replay size (enables is_ready property, when fills).
        alpha (float): Exponent which determines how much priority is used.
            (0 - uniform prioritization, 1 - full prioritization).
        beta (float): Exponent which determines how much importance-sampling correction is used.
            (0 - no correction, 1 - full correction).
    """
    def __init__(self, capacity, batch_size, min_size=0, alpha=0.7, beta=0.5):
        super(ProportionalReplay, self).__init__(capacity, batch_size, min_size)
        assert alpha >= 0
        assert beta >= 0
        self.sumtree = SumTree(capacity)
        self.mintree = MinTree(capacity)
        self._alpha = alpha
        self._beta = beta
        self._epsilon = 0.00001
        self._max_priority = 0.0

    def _preproc_priority(self, error):
        return (error + self._epsilon) ** self._alpha

    def add(self, obs, action, reward, term, obs_next, priority=None):
        if priority is None:
            priority = self._max_priority
        super(ProportionalReplay, self).add(obs, action, reward, term, obs_next)
        self.sumtree.append(self._preproc_priority(priority))
        self.mintree.append(self._preproc_priority(priority))

    def sample(self):
        idxs = []
        proportion = self.sumtree.sum() / self._batch_size
        for i in range(self._batch_size):
            sum_from = proportion * i
            sum_to = proportion * (i + 1)
            s = random.uniform(sum_from, sum_to)
            idxs.append(self.sumtree.find_sum_idx(s))
        gather = itemgetter(*idxs)
        next_obs_gather = itemgetter(*[i + 1 for i in idxs])
        importances = self._compute_importance(idxs, self._beta)
        return (gather(self._obs),
                gather(self._actions),
                gather(self._rewards),
                gather(self._terms),
                next_obs_gather(self._obs),
                np.ones_like(idxs, 'bool'),
                idxs,
                importances)
        # traj = TrajectoryBatch(obses=gather(self._obs),
        #                        actions=gather(self._actions),
        #                        rewards=gather(self._rewards),
        #                        terms=gather(self._terms),
        #                        next_obses=next_obs_gather(self._obs),
        #                        ends=np.ones_like(idxs, 'bool'))
        # return traj, idxs, importances

    def _compute_importance(self, indexes, beta):
        importances = [0.0] * len(indexes)
        if self.mintree.min() == float('inf'):
            return importances
        prob_min = self.mintree.min() / self.sumtree.sum()
        weight_max = (prob_min * self.sumtree.size) ** (-beta)
        for i, idx in enumerate(indexes):
            prob = self.sumtree[idx] / self.sumtree.sum()
            weight = (prob * self.sumtree.size) ** (-beta)
            importances[i] = weight / weight_max
        return importances

    def update(self, indexes, priorities):
        if not isinstance(priorities, np.ndarray):
            priorities = np.asarray(priorities)
        priorities += self._epsilon
        priorities = self._preproc_priority(priorities)
        for idx, prior in zip(indexes, priorities):
            self._max_priority = max(self._max_priority, prior)
            self.sumtree.update(int(idx), prior)


class BackPropagationReplay(ExperienceReplay):
    def __init__(self, capacity, batch_size,
                 accum_initial, accum_func,
                 min_size=0, accum_bias=0, beta=10, lambd=3):
        super(BackPropagationReplay, self).__init__(capacity, batch_size, min_size)
        self._beta = beta
        self.sumtree = SumTree(capacity)
        self._lambd=lambd
        self._timestamp_counter = 0
        self._timestamp = [-1, ] * capacity
        self._origins = [-1, ] * capacity
        self._factor = [1, ] * capacity
        self._priority = [1, ] * capacity
        self._accum_initial = accum_initial
        self._accum_bias = accum_bias
        self._accum = accum_initial
        self._accum_func = accum_func

    # This algorithm requires only clipped reward
    def _preproc_priority(self, reward):
        return self._accum_func(self._accum, reward) + self._accum_bias

    def add(self, obs, action, reward, term, obs_next):
        idx = self._idx
        prev_factor = self._factor[idx]
        prev_origin = self._origins[idx]
        super(BackPropagationReplay, self).add(obs, action, reward, term, obs_next)
        self._factor[idx] = self._beta if reward != 0 else 1.
        self._priority[idx] = self._preproc_priority(reward)
        self._timestamp[idx] = self._timestamp_counter
        self.sumtree.append(self._priority[idx] * self._factor[idx])

        if reward != 0 or term:
            self._origins[idx] = idx

        if term:
            self._accum = self._accum_initial

        # If history is removed, update predecessor chains
        next_idx = self._idx
        if self._timestamp[next_idx] != -1 and idx != prev_origin and prev_factor > 1:
            self._factor[next_idx] = prev_factor
            self.sumtree.update(next_idx, self._factor[next_idx] * self._priority[next_idx])

        self._timestamp_counter += 1

    def sample(self):
        idxs = []
        proportion = self.sumtree.sum() / self._batch_size
        for i in range(self._batch_size):
            sum_from = proportion * i
            sum_to = proportion * (i + 1)
            s = random.uniform(sum_from, sum_to)
            idxs.append(self.sumtree.find_sum_idx(s))
        gather = itemgetter(*idxs)
        next_obs_gather = itemgetter(*[i + 1 for i in idxs])
        # After sampling propagate its value
        for idx in idxs:
            predecessor = self._cycle_idx(idx - 1)
            if self._timestamp[predecessor] != -1 \
                    and self._timestamp[predecessor] < self._timestamp[idx]:

                if (not self._terms[predecessor]
                        and self._rewards[predecessor] == 0
                        and self._factor[idx] > 1):
                    self._factor[predecessor] = self._factor[idx]
                    self._origins[predecessor] = self._origins[idx]
                    self.sumtree.update(predecessor, self._factor[predecessor] * self._priority[predecessor])

                elif self._factor[idx] > 1:
                    origin = self._origins[idx]
                    factor_origin = self._factor[idx]
                    self._factor[origin] = max(1, factor_origin // self._lambd)
                    self.sumtree.update(origin, self._factor[origin] * self._priority[origin])

                self._factor[idx] = 1
                self.sumtree.update(idx, self._factor[idx] * self._priority[idx])

        return (gather(self._obs),
                gather(self._actions),
                gather(self._rewards),
                gather(self._terms),
                next_obs_gather(self._obs),
                np.ones_like(idxs, 'bool'),
                idxs,
                [1.0] * len(idxs))


class WindoedBackPropagationReplay(ExperienceReplay):
    def __init__(self, capacity, batch_size, window_size,
                 min_size=0, beta=10, lambd=3):
        super(WindoedBackPropagationReplay, self).__init__(capacity, batch_size, min_size)
        self._beta = beta
        self.sumtree = SumTree(capacity)
        self._lambd=lambd
        self._timestamp_counter = 0
        self._timestamp = [-1, ] * capacity
        self._origins = [-1, ] * capacity
        self._factor = [1, ] * capacity
        self._counter = [1, ] * capacity
        self._window_size = window_size

    # This algorithm requires only clipped reward
    def _preproc_priority(self, counter):
        if counter == 0:
            return 1
        return 2 - 1 / 2 ** (counter - 1)

    def add(self, obs, action, reward, term, obs_next):
        # If reward is not zero update previous experiences counter
        idx = self._idx
        prev_factor = self._factor[idx]
        prev_counter = self._counter[idx]
        prev_origin = self._origins[idx]
        for i in reversed(range(1, self._window_size + 1)):
            descent_experience = self._cycle_idx(self._idx - i)
            if self._timestamp[descent_experience] <= self._timestamp_counter:
                self._counter[descent_experience] += 1
                self.sumtree.update(
                    descent_experience,
                    self._factor[descent_experience] * self._preproc_priority(self._counter[descent_experience]))

        super(WindoedBackPropagationReplay, self).add(obs, action, reward, term, obs_next)
        self._factor[idx] = self._beta if reward != 0 else 1.
        self._counter[idx] = 1 if reward != 0 else 0
        self._timestamp[idx] = self._timestamp_counter
        self.sumtree.append(self._counter[idx] * self._preproc_priority(self._factor[idx]))

        if reward != 0 or term:
            self._origins[idx] = idx

        # If history is removed, update predecessor chains
        next_idx = self._idx
        if self._timestamp[next_idx] != -1 and prev_origin != idx:
            need_update = False
            if prev_factor > 1:
                self._factor[next_idx] = prev_factor
                need_update = True
            if prev_counter > 1:
                self._counter[next_idx] = prev_counter
                need_update = True

            if need_update:
                self.sumtree.update(
                    next_idx,
                    self._factor[next_idx] * self._preproc_priority(self._counter[next_idx]))

        self._timestamp_counter += 1

    def sample(self):
        idxs = []
        proportion = self.sumtree.sum() / self._batch_size
        for i in range(self._batch_size):
            sum_from = proportion * i
            sum_to = proportion * (i + 1)
            s = random.uniform(sum_from, sum_to)
            idxs.append(self.sumtree.find_sum_idx(s))
        gather = itemgetter(*idxs)
        next_obs_gather = itemgetter(*[i + 1 for i in idxs])
        # After sampling propagate its value
        for idx in idxs:
            predecessor = self._cycle_idx(idx - 1)
            if predecessor != -1 and self._timestamp[predecessor] < self._timestamp[idx]:
                if (not self._terms[predecessor]
                        and self._rewards[predecessor] == 0
                        and self._factor[idx] > 1):
                    self._factor[predecessor] = self._factor[idx]
                    self._counter[predecessor] = self._counter[idx]
                    self._origins[predecessor] = self._origins[idx]
                    self.sumtree.update(
                        predecessor,
                        self._factor[predecessor] * self._preproc_priority(self._counter[predecessor]))

                elif self._factor[idx] > 1:
                    origin = self._origins[idx]
                    factor_origin = self._factor[idx]
                    counter_origin = self._counter[idx]
                    self._counter[origin] = counter_origin
                    self._factor[origin] = max(1, factor_origin // self._lambd)
                    self.sumtree.update(
                        origin,
                        self._factor[origin] * self._preproc_priority(self._counter[origin]))

                self._factor[idx] = 1
                self._counter[idx] = 1
                self.sumtree.update(idx, self._factor[idx] * self._counter[idx])

        return (gather(self._obs),
                gather(self._actions),
                gather(self._rewards),
                gather(self._terms),
                next_obs_gather(self._obs),
                np.ones_like(idxs, 'bool'),
                idxs,
                [1.0] * len(idxs))

