from core import base_environment
from gym import spaces

class GymWrapper(object):
  def __init__(self, env):
    self._env = env
    #self.action_space = spaces.Discrete(self.env.num_actions())
    self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
    obsNdim = self.env.observation_ndim()   
    obsKeys = obsNdim.keys()
    assert len(obsKeys) == 1, 'gym only supports one observation type'
    self._obsKey = obsKeys[0]
    self.observation_space = spaces.Box(low=0, high=255, shape=obsNdim[obsKeys[0]])

  @property
  def frameskip(self):
    raise NotImplementedError

  @property
  def ale(self):
    raise NotImplementedError

  @property
  def env(self):
    return self._env

  @property
  def _n_actions(self):
    return self.env.action_dim()

  def _reset(self):
    self.env.reset()
    return self.env.observation()

  def reset(self):
    return self.reset()

  def _step(self, action):
    assert type(action) == np.ndarray, 'action must be a nd-array'
    self.env.step(action)
    obs    = self.env.observation()
    reward = self.env.reward()
    done   = False
    return obs, reward, done, dict(reward=reward)

  def step(self, action):
    return self._step(action)

  def _get_obs(self):
    return self.env.observation()[self._obsKey]
 
  def viewer_setup(self):
    self.env._renderer_setup() 

  def render(self):
    return self.render()

  def _render(self):
    return self.env.render() 
  
