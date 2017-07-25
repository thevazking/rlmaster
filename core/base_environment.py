import abc
import copy
import numpy as np
import os
from os import path as osp
from pyhelper_fns import vis_utils as vu
from pyhelper_fns import check_utils

class BaseObject(object):
  def __init__(self, simulator, prms={}):
    updatedPrms      = self.defaultPrms
    updatedPrms.update(prms)
    self._prms = updatedPrms

  @property
  def prms(self):
    return self._prms

  @property
  def defaultPrms(self):
    return {}

  def get(self):
    raise NotImplementedError


class BaseObjectWithSim(BaseObject):
  def __init__(self, simulator, prms={}):
    super(BaseObjectWithSim, self).__init__(prms)
    self._simulator  = simulator

  @property
  def simulator(self):
    return self._simulator

    
##
#Base class for defining the observation that the agent receives
class BaseObservation(BaseObjectWithSim):
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
class BaseInitializer(BaseObjectWithSim):
  def __init__(self, simulator, prms={}):
    super(BaseInitializer, self).__init__(simulator, prms=prms)
    self._random = np.random.RandomState(self.prms['randomSeed'])

  @property
  def random(self):
    return self._random

  @property
  def defaultPrms(self):
    defPrms = {}
    defPrms['randomSeed'] = 3
    return defPrms

  def sample_env_init(self):
    raise NotImplementedError


##
#Base class for defining reward functions.
class BaseRewarder(BaseObjectWithSim):
  pass 

##
#BaseAction class
class BaseAction(object):
  __metaclass__ = abc.ABCMeta
  def __init__(self, prms={}):
    self._prms = prms
   
  @abc.abstractmethod 
  def action_dim(self):
    """
    Returns:
      The dimensionality of the input action 
      eg: for 1 out N discrete action, action_dim=1
          if there are 3 joint angles, action_dim=3
      
    """

  @abc.abstractmethod
  def num_actions(self):
    """
    Returns:
      Number of different possible actions
      eg: for 1 out N discrete action, num_actions=3
          if there are 3 joint angles, num_actions=1
    """

  @abc.abstractmethod
  def minval(self):
    """
    Returns:
      for discrete_action: the value of lowest action
      for continous_action: the minimum value of action
                            in any dimension
    """

  @abc.abstractmethod
  def maxval(self):
    """
    Returns:
      for discrete_action: the value of highest action
      for continous_action: the maximum value of action
                            in any dimension
    """

  #Process the action as needed
  @abc.abstractmethod
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
class BaseSimulator(BaseObject):
  """
  Setup the object that simulates the environment (for eg: the physics engine.)
  """
  def __init__(self, prms={}, isRender=False):
    super(BaseSimulator, self).__init__(prms)
    self._isRender  = isRender 
    
  @property
  def defaultPrms(self):
    defPrms = super(BaseSimulator, self).defaultPrms
    defPrms['actionRepeat'] = 4
    return defPrms

  @property
  def isRender(self):
    return self._isRender

  @isRender.setter
  def isRender(self, isRender):
    self._isRender = isRender

  @property
  def simParams(self):
    return copy.deepcopy(self.prms)

  def _setup_world(self):
    pass

  def step(self, ctrl):
    raise NotImplementedError

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


BASE_ENV_PARAMS = {
  'action_repeat': 1,
  'max_episode_length': 100
}

class BaseEnvironment(object):
  def __init__(self, sim, initializer, observer, rewarder,
               action_processor, params={}):
    self._simulator   = sim
    self._initializer = initializer
    self._observer    = observer
    self._rewarder    = rewarder
    self._action_processor = action_processor
    self.params       = copy.deepcopy(BASE_ENV_PARAMS)
    for k in params.keys():
      assert k in BASE_ENV_PARAMS.keys(), '%s not found' % k
    self.params.update(params)
    self.stepCount    = 0
    self.reset()

  @property
  def simulator(self):
    return self._simulator

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
    self.stepCount = 0


  def step(self, ctrl):
    """
    Step the simulator by 1 step using ctrl command
    """
    if self.stepCount <= self.params['max_episode_length']:
      ctrl = self.action_processor.process(ctrl)
      for n in range(self.params['action_repeat']):
        self.simulator.step(ctrl)
      self.stepCount += 1
    else:
      raise Exception('Maximum Episode Length exceeded')


  def step_by_n(self, ctrl, N):
    """
    Step the simulator by N time steps
    """
    for n in range(N):
      self.step(ctrl)


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


  def step_nd_render(self, ctrl, N=1):
    self.step_by_n(ctrl, N)
    self.render() 


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


def visualize_random_exploration(env, mode='randSample', 
              numEpisodes=10, episodeLength=500):
  anim = vu.MyAnimation(None)
  if isinstance(env, BaseEnvironment):
    for e in range(10):
      _  = env.reset()
      for i in range(episodeLength):
        if mode == 'randSample':
          env.step(0.6 * np.random.randn(env.simulator.num_actuators(),))
        else:
          env.step(0.4 * np.zeros(env.simulator.num_actuators(),))
        im = env.simulator.get_image(cName='main')
        anim._display(im)
  else:
    raise Exception('Invalid argument type {0}'.format(type(env)))


def save_random_exploration_video(env, mode='randSample', 
              numEpisodes=10, episodeLength=500):
  vidName = 'random_exploration.mp4'
  vid     = vu.VideoMaker()
  if isinstance(env, BaseEnvironment):
    for e in range(numEpisodes):
      _  = env.reset()
      for i in range(episodeLength):
        if mode == 'randSample':
          env.step(0.6 * np.random.randn(env.simulator.num_actuators(),))
        else:
          env.step(0.4 * np.zeros(env.simulator.num_actuators(),))
        im = env.simulator.get_image(cName='main')
        vid.save_frame(im)
  else:
    raise Exception('Invalid argument type {0}'.format(type(env)))
  vid.compile_video()


