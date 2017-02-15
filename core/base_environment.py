import copy
import numpy as np
import os
from os import path as osp
import vis_utils as vu
from pyhelper_fns import check_utils

from mujoco_py_wrapper.mujoco_agents.config import AGENT_MUJOCO
from mujoco_py_wrapper import mujoco_py as mjcpy

class BaseObject(object):
  def __init__(self, env, prms=None):
    self._env  = env
    self._prms = prms

  @property
  def env(self):
    return env

  @property
  def prms(self)
    return prms

  def get(self):
    raise NotImplementedError
    

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

class BaseInitializer(BaseObject):
  def __init__(self, env, **kwargs):
    super(BaseObject, env).__init__(env, **kwargs)
    seed = self.prms['randomSeed'] if hasattr(self.prms, 'randomSeed') else 3
    self._random = np.random.RandomState(seed)

  @property
  def random(self):
    return self._random

  def sample_env_init(self):
    pass

class BaseRewarder(BaseObject):
  pass 


class BaseSimulator(object):
  """
  Setup the object that simulates the environment (for eg: the physics engine.)
  """
  def __init__(self, simParams=None, isRender=False):
    self._simParams = simParams
    self._isRender  = isRender 
    
  @property
  def is_render(self):
    return self._isRender

  @is_render.setter
  def is_render(self, is_render):
    self._isRender = render

  def _setup_world(self):
    pass 

  def step(self, ctrl):
    pass

  def step_by_n(self, N, ctrl):
    for n in range(N):
      self.step(ctrl)

  def render(self):
    pass

  def get_image(self):
    pass


class BaseEnvironment(object):
  def __init__(self, sim, initializer, observer, rewarder, params=None):
    self._sim = sim
    self._initializer = initializer
    self._observer    = observer
    self._rewarder    = rewarder
    self.params      = params 

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

  def action_ndim(self):
    """
    Return the dimensionality of the actions
    """
    pass

  def step(self, ctrl):
    """
    Step the simulator by 1 step using ctrl command
    """
    self._sim.step(self, ctrl)

  def step_by_n(self, N, ctrl):
    """
    Step the simulator by N time steps
    """
    self._sim.step_by_n(self, ctrl)

  def observation(self):
    """
    Observe the environment's state
    """ 
    return self._observer.observation()

  def observation_ndim(self):
    """
    Dimensionality of observation
    """ 
    return self._observer.observation_ndim()

  def reset(self):
    """
    Reset the environment
    """
    self._initializer.sample_env_init() 

  def reward(self):
    """
    Return the reward
    """
    return self._rewarder.get()

  def interactive(self, str2action):
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
      else:
        ctrl = str2action(ip)
        if ctrl is None:
          continue
        else:
          self.step_by_n(5, ctrl)
      self.render()


class BaseMujoco(object):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams, is_render=False):
      config = copy.deepcopy(AGENT_MUJOCO)
      config.update(hyperparams)
      self._hyperparams = config
      #Setup the mujoco environment
      self._setup_world(hyperparams['filename'])
      #Rendering flag
      self._is_render = is_render

    def __del__(self):
      self._small_viewer.finish()

    @property
    def model(self):
      return self._model
  
    
    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        self._model = mjcpy.MjModel(filename)
        #self._joint_idx = list(range(self._model['nq']))
        #self._vel_idx = [i + self._model['nq'] for i in self._joint_idx]
        self._setup_viewer()


    def _setup_viewer(self):
        if self._is_render:
          cam_pos = self._hyperparams['camera_pos']
          gofast = True
          self._small_viewer = mjcpy.MjViewer(visible=True,
                                              init_width=AGENT_MUJOCO['image_width'],
                                              init_height=AGENT_MUJOCO['image_height'],
                                              go_fast=gofast)
          self._small_viewer.start()
          self._small_viewer.set_model(self._model)
          self.render()


    def render(self):
      self._small_viewer.render()
      self._small_viewer.loop_once()


    def step(self, ctrl=None):
      if ctrl is not None:
        self._model.data.ctrl = copy.deepcopy(ctrl)
      self.model.step()

    def step_by_n(self, N, ctrl=None):
      for n in range(N):
        self.step(ctrl)


    def get_image(self):
      img_string, width, height = self._small_viewer.get_image()
      img = np.fromstring(img_string, dtype='uint8').reshape(
            (height, width, self._hyperparams['image_channels']))[::-1, :, :]
      return img


    def interactive(self):
      while True:
        isValid = False
        ip      = raw_input()
        if ip == 'q':
          break
        else:
          cmds = ip.split(',')
          if not len(cmds) == 2:
            continue
          else:
            ctrl = np.array([float(cmds[0]), float(cmds[1])])
        self.step_by_n(5, ctrl)
        self.render()


    def observation(self):
      """
      Returns:
        obs: a dict of agent's observations
      """
      raise NotImplementedError

    def log_diagnostics(self):
      raise NotImplementedError  



def simple_test():
  hyperparams = {}
  #DATA_DIR   = osp.join(os.getenv('HOME'), 'code', 'gps', 'mjc_models') 
  DATA_DIR    = osp.join('/work4/pulkitag-code/pkgs/','gps', 'mjc_models') 
  hyperparams['filename'] = osp.join(DATA_DIR, 'particle2d.xml')
  check_utils.exists(hyperparams['filename']) 
  ag = BaseAgent(hyperparams)
  return ag 
  ctrl = np.array(([1., 1.]))
  print (ag.model.data.xpos)
  ag.render()
  im1  = ag.get_image()
  ag.step_by_n(1000, ctrl)
  print (ag.model.data.xpos)
  ag.render()
  im2  = ag.get_image()
  vu.plot_pairs(im1, im2) 
  return ag
