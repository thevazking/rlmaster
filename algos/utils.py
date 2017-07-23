import numpy as np

class ReplayBufferBasic(object):
  """Stores (o_tm1, a_tm1, o_t, r_t) tuples 
     The replay buffer is in it's current form is not
     parallelizable. The reason is two fold:
     (1) The experiences are stored in four different lists instead of one list
         this means that these lists need to be concurrently updated
     (2) After max. buffer size is reached, new experiences are overwritten on the list
         using a list index. This means that index needs to be in sync on workers.
  """
  def __init__(self, minSz=1000, maxSz=10000, randSeed=3):
    #Observation at time step t-1
    self._o_tm1 = []
    #Action at time t-1
    self._a_tm1 = []
    #Observation at time step t
    self._o_t = []
    #Reward
    self._r_t = []
    #Terminal
    self._end = []
    self.N = 0
    self.idx = 0
    self.minSz = minSz
    self.maxSz = maxSz
    self._rand = np.random.RandomState(randSeed)
    self._rand_count = 0
    self._perm = []

  def size(self):
    return self.N

  def get_sample_index(self, batchSz):
    if self.size() > len(self._perm) or self._rand_count + batchSz > self.size():
      self._perm = self._rand.permutation(self.size())
      self._rand_count = 0
    st = self._rand_count
    en = st + batchSz
    self._rand_count += batchSz
    return self._perm[st:en]

  def get_ordered_sample_index(self, batchSz):
    if self._rand_count + batchSz > self.N:
      idx1 = range(self._rand_count, self.N)
      idx2 = range(0, batchSz - len(idx1))
      self._rand_count = len(idx1)
      idxs = idx1 + idx2
    else:
      idxs= range(self._rand_count, self._rand_count + batchSz)
      self._rand_count += batchSz
    return idxs

  def add(self, data):
    if len(data) == 4:
      o_tm1, a_tm1, o_t, r_t = data
      _isEnd = False
    else:
      o_tm1, a_tm1, o_t, r_t, isEnd = data
    if self.size() >= self.maxSz:
      self._o_tm1.pop(0)
      self._a_tm1.pop(0)
      self._o_t.pop(0)
      self._r_t.pop(0)
      self._end.pop(0)
      self._o_tm1.append(o_tm1.copy())
      self._a_tm1.append(a_tm1.copy())
      self._o_t.append(o_t.copy())
      self._r_t.append(r_t.copy())
    else:
      self._o_tm1.append(o_tm1.copy())
      self._a_tm1.append(a_tm1.copy())
      self._o_t.append(o_t.copy())
      self._r_t.append(r_t.copy())
      self.N += 1

  def _stack(self, x, idx):
    return np.vstack([x[i].copy() for i in idx])

  def _sample_from_index(self, idxs):
    return (self._stack(self._o_tm1, idxs), 
            self._stack(self._a_tm1, idxs),
            self._stack(self._o_t, idxs), 
            self._stack(self._r_t, idxs))

  def sample_in_order(self, batchSz):
    """Ordered sampling from the buffer"""
    idxs = self.get_ordered_sample_index(batchSz)
    return self._sample_from_index(idxs) 

  def sample(self, batchSz):
    """ Random sampling from the buffer """
    idxs = self.get_sample_index(batchSz)
    return self._sample_from_index(idxs) 


class Episode(object):
  """
  Note that this implementation implicity
  assumes that we are taking samples from a infinite
  horizon episodes
  """
  def __init__(self, rand_seed=None):
    self._o_tm1 = {}
    self._a_tm1 = []
    self._r_t   = []
    self._is_terminal = []
    self._gamma = []
    self.N = 0
    self.ready = False
    self._perm = []
    self._rand_count = 0
    if rand_seed is None:
      self._rand = np.random
    else:
      self._rand = np.random.RandomState(rand_seed)

  def size(self):
    return self.N

  def get_sample_index(self):
    if self._rand_count + 1 > self.size() or len(self._perm) < self.size():
      self._perm = self._rand.permutation(self.size())
      self._rand_count = 0
    idx = self._perm[self._rand_count]
    self._rand_count += 1
    return idx
  
  def get_sample_index_order(self):
    if self._rand_count + 1 > self.size():
      self._rand_count = 0
    idx = self._rand_count
    self._rand_count += 1
    return idx

  def add(self, data):
    """
    Args:
     o_tm1: dict of observation keys
            o_tm1[im]: 1 x h x w x 3
     r_t  : array of size 1 x 1
    """
    assert not self.ready, 'the episode is already finalized'
    o_tm1, a_tm1, r_t, is_terminal, gamma = data
    #If the first bit of data is being added
    if len(self._o_tm1.keys()) == 0:
      for k in o_tm1.keys():
        self._o_tm1[k] = []
    #Add data
    for k in o_tm1.keys():
      assert o_tm1[k].shape[0] == 1, 'First dimension is the batch, should be 1'
      self._o_tm1[k].append(o_tm1[k].copy())
    assert a_tm1.shape[0] == 1
    assert r_t.shape[0] == 1
    self._a_tm1.append(a_tm1.copy())
    self._r_t.append(r_t.copy())
    self._is_terminal.append(is_terminal.copy())
    self._gamma.append(gamma.copy())
    self.N += 1

  def finish(self):
    #Remove the last action and reward because they donot
    #have a next observation
    if not self._is_terminal[-1][0][0]:
      #The episode is assumed to be infinite length
      self._a_tm1.pop(-1)
      self._r_t.pop(-1)
    self.N -= 1
    self.ready = True

  def _append_obs_into_channels(self, x, idx):
    """
    concatenate by the last dimension
    """
    obs_list = []
    for i in idx:
      if i >= len(self._is_terminal):
        print (i, len(self._is_terminal), self.N)
      if self._is_terminal[i][0][0]:
        obs_list.append(x[i-1].copy())
      else:
        obs_list.append(x[i].copy())
    return np.concatenate(obs_list, len(x[0].shape)-1)

  def _stack_obs(self, idx, concat={}):
    """
      concat: dict specificying which observation
              should be stacked by how much
              for eg:{'im': 2} means stack
              previous two frames
    """
    obs = {}
    for k in self._o_tm1.keys():
      if k in concat.keys():
        st = idx - concat[k] + 1
        en = idx
        rng = range(st, en+1)
      else:
        rng = range(idx, idx+1)
      obs[k] = self._append_obs_into_channels(self._o_tm1[k], rng)
    return obs

  def _sample_from_index(self, idx, concat={}):
    o_tm1 = self._stack_obs(idx, concat)
    o_t   = self._stack_obs(idx+1, concat)
    a_tm1 = self._a_tm1[idx].copy()
    r_t   = self._r_t[idx].copy()
    gamma   = self._gamma[idx+1].copy()
    return (o_tm1, a_tm1, o_t, r_t, gamma)

  def sample(self, concat={}):
    """
      concat: dict specificying which observation
              should be stacked by how much
    """
    assert self.ready
    idx = self.get_sample_index()
    assert idx < self.N
    return self._sample_from_index(idx, concat)

  def sample_in_order(self, concat={}):
    assert self.ready
    idx = self.get_sample_index_order()
    assert idx < self.N
    return self._sample_from_index(idx, concat)



class ReplayBufferEpisodic(object):
  """
    Store experiences as a list of episodes
  """
  def __init__(self, minSz=1000, maxSz=10000, randSeed=3, concat={}):
    """
      minSz: minimum number of examples
      maxSz: maximum number of examples
    """
    self.num_episodes = 0
    self.num_samples  = 0
    self.minSz = minSz
    self.maxSz = maxSz
    self._episodes = []
    self.randSeed = randSeed
    self._rand    = np.random.RandomState(randSeed)
    self._current_episode = None
    self.concat = concat

  def size(self):
    return self.num_samples

  def begin_episode(self):
    while (self.num_samples >= self.maxSz):
      eps = self._episodes.pop(0)
      self.num_samples -= (eps.size())
      self.num_episodes -= 1
    self._current_episode = Episode()

  def add(self, data):
    self._current_episode.add(data)

  def end_episode(self):
    self._current_episode.finish()
    if self._current_episode.N > 1:
      self.num_samples += self._current_episode.size()
      self._episodes.append(self._current_episode)
      self.num_episodes += 1
    else:
      logging.info('Ignoring episode as it is of length 1')

  def _append_obs(self, obs, odict):
    if len(odict.keys()) == 0:
      for k in obs.keys():
        odict[k] = []
    for k, v in obs.iteritems():
      odict[k].append(v)

  def sample(self, batchSz):
    """
    Produce a random sample in the following manner:
    # Random sample a subset of episodes
    # From this random sample - sample random interaction
      from the environment
    Args:
      batchSz: the number of interactions to sample
    Returns:
      all_o_tm1: dict of observations where
                 all_o_tm1[im]: np.array of size batchSz, h, w, 3
                 .. similarly other observations
      all_a_tm1 : np.array of size batchSz x action_dimension
      all_o_t:  same but at time step t
      r_t, gamma
     
     
    """
    all_a_tm1, all_r_t = [], []
    all_gamma  = []
    all_o_tm1, all_o_t = {}, {}
    for b in range(batchSz):
      eps = int(np.floor(self._rand.rand() * (self.num_episodes -1)))
      o_tm1, a_tm1, o_t, r_t, gamma = self._episodes[eps].sample(concat=self.concat)
      self._append_obs(o_tm1, all_o_tm1)
      self._append_obs(o_t,   all_o_t)
      all_a_tm1.append(a_tm1)
      all_r_t.append(r_t)
      all_gamma.append(gamma)
    for k in all_o_tm1.keys():
      all_o_tm1[k] = np.vstack(all_o_tm1[k])
      all_o_t[k]   = np.vstack(all_o_t[k])
    all_a_tm1 = np.vstack(all_a_tm1)
    all_r_t   = np.vstack(all_r_t)
    all_gamma = np.vstack(all_gamma)
    return all_o_tm1, all_a_tm1, all_o_t, all_r_t, all_gamma
