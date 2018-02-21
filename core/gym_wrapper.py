import numpy as np
from core import base_environment
from gym import spaces
from gym.utils import seeding

#class GymWrapper(gym.Wrapper):
class GymWrapper(object):
  def __init__(self, env):
    self._env = env
    if isinstance(self.env.action_processor, 
                  base_environment.BaseDiscreteAction):
      self.action_space = spaces.Discrete(self.env.num_actions())
    else:
      assert isinstance(self.env.action_processor,
                        base_environment.BaseContinuousAction)
      self.action_space = spaces.Box(
                      low  = env.action_processor.minval(), 
                      high = env.action_processor.maxval(), 
                      shape=(env.action_processor.action_dim(), 
                             env.action_processor.num_actions()))
    obsNdim = self.env.observation_ndim()   
    obsKeys = obsNdim.keys()
    obsKeys = list(obsKeys) # python3 compatibility
    assert len(obsKeys) == 1, 'gym only supports one observation type'
    self._obsKey = obsKeys[0]
    self.observation_space = spaces.Box(low=0, high=255, shape=obsNdim[obsKeys[0]])

  # New gym compatibility
  metadata = {'render.modes': []}
  reward_range = (-np.inf, np.inf)
  spec = None

  @property
  def unwrapped(self):
      return self

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

  def _observation(self):
    """
    Gym environment only supports one kind of observation
    """
    return self.env.observation()[self._obsKey]


  def _reset(self):
    self.env.reset()
    return self._observation()


  def reset(self):
    return self._reset()


  def _step(self, action):
    assert type(action) == np.ndarray, 'action must be a nd-array'
    self.env.step(action)
    obs    = self._observation()
    reward = self.env.reward()
    done   = False
    return obs, reward, done, dict(reward=reward)

  # new gym compatibility
  def _seed(self):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]

  def step(self, action):
    return self._step(action)


  def viewer_setup(self):
    self.env._renderer_setup() 


  def render(self):
    return self._render()

  # mode is hack, doesn't actually do anything
  def _render(self, mode, close=False):
    return self.env.render() 
  
