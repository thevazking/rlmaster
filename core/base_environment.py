import copy
import numpy as np
import os
from os import path as osp
from pyhelper_fns import vis_utils as vu
from pyhelper_fns import check_utils

class BaseObject(object):
  def __init__(self, simulator, prms={}):
    self._simulator  = simulator
    self._prms = prms

  @property
  def simulator(self):
    return self._simulator

  @property
  def prms(self):
    return self._prms

  def get(self):
    raise NotImplementedError
    

##
#Base class for defining the observation that the agent receives
class BaseObservation(BaseObject):
  def spec(self):
    """
    Returns the observation in the format
    required by the agent
    """
    pass

  def ndim(self):
    """
    Returns the dimensionality of flattened observation
    if there is a single observation type, else a dict. 
    """
    pass

  def observation(self):
    """
    The observation used by the agent
    Returns:
      a dict of observation
    """
    pass


##
#Base class for defining how the environment should be initialized at every
#episode
class BaseInitializer(BaseObject):
  def __init__(self, simulator, prms={}):
    super(BaseInitializer, self).__init__(simulator, prms=prms)
    seed = self.prms['randomSeed'] if hasattr(self.prms, 'randomSeed') else 3
    self._random = np.random.RandomState(seed)

  @property
  def random(self):
    return self._random

  def sample_env_init(self):
    raise NotImplementedError


##
#Base class for defining reward functions.
class BaseRewarder(BaseObject):
  pass 

##
#BaseAction class
class BaseAction(object):
  def __init__(self, prms={}):
    self._prms = prms
    
  def action_dim(self):
    """
    Returns:
      The dimensionality of the input action 
      eg: for 1 out N discrete action, action_dim=1
          if there are 3 joint angles, action_dim=3
      
    """
    raise NotImplementedError

  def num_actions(self):
    """
    Returns:
      Number of different possible actions
      eg: for 1 out N discrete action, num_actions=3
          if there are 3 joint angles, num_actions=1
    """
    raise NotImplementedError

  #Process the action as needed
  def process(self, action):
    """
    Args:
      action: input from the agent
    """
    raise NotImplementedError


class BaseDiscreteAction(BaseAction):
  def action_dim(self):
    return 1


class BaseContinuousAction(BaseAction):
  def num_actions(self):
    return 1
  

##
#Base class for defining environment simulation.
class BaseSimulator(object):
  """
  Setup the object that simulates the environment (for eg: the physics engine.)
  """
  def __init__(self, simParams=None, isRender=False):
    self._simParams = simParams
    self._isRender  = isRender 
    
  @property
  def isRender(self):
    return self._isRender

  @isRender.setter
  def isRender(self, isRender):
    self._isRender = isRender

  @property
  def simParams(self):
    return copy.deepcopy(self._simParams)

  def _setup_world(self):
    pass

  def step(self, ctrl):
    raise NotImplementedError

  def step_by_n(self, N, ctrl):
    for n in range(N):
      self.step(ctrl)

  def _setup_renderer(self):
    """
    Setup the renderer
    """
    raise NotImplementedError

  def render(self):
    """
    create the rendering of the environment 
    """
    raise NotImplementedError

  def get_image(self):
    raise NotImplementedError


class BaseEnvironment(object):
  def __init__(self, sim, initializer, observer, rewarder,
               action_processor, params=None):
    self._sim = sim
    self._initializer = initializer
    self._observer    = observer
    self._rewarder    = rewarder
    self._action_processor = action_processor
    self.params       = params 
    self.reset()

  @property
  def simulator(self):
    return self._sim

  @property
  def observer(self):
    return self._observer

  @property
  def rewarder(self):
    return self._rewarder

  @property
  def initializer(self):
    return self._initializer

  @property
  def action_processor(self):
    return self._action_processor 


  def action_dim(self):
    """
    Return the dimensionality of the actions
    """
    return self.action_processor.action_dim()


  def num_actions(self):
    """
    Return the number of possible actions
    """
    return self.action_processor.num_actions()


  def reset(self):
    """
    Reset the environment
    """
    self.initializer.sample_env_init() 


  def step(self, ctrl):
    """
    Step the simulator by 1 step using ctrl command
    """
    ctrl = self.action_processor.process(ctrl)
    self.simulator.step(ctrl)


  def step_by_n(self, N, ctrl):
    """
    Step the simulator by N time steps
    """
    self.simulator.step_by_n(ctrl)


  def observation(self):
    """
    Observe the environment's state
    """ 
    return self.observer.observation()


  def observation_ndim(self):
    """
    Dimensionality of observation
    """ 
    return self.observer.ndim()


  def reward(self):
    """
    Return the reward
    """
    return self.rewarder.get()


  def setup_renderer(self):
    """
    Setup the rendering mechanism
    """
    self.simulator._setup_renderer()


  def render(self):
    """
    Render the environment
    """
    self.simulator.render()


  def interactive(self, str2action, actionRepeat=None):
    """
    Interact with the environment
    Args:
      str2action: a function that takes in a string
                  and converts them into action
    """
    while True:
      isValid = False
      ip      = raw_input()
      if ip == 'q':
        break
      elif ip == 'reset':
        self.reset()
      else:
        ctrl = str2action(ip)
        if ctrl is None:
          continue
        else:
          if actionRepeat is None:
            self.step(ctrl)
          else:
            self.step_by_n(actionRepeat, ctrl)
      self.render()



