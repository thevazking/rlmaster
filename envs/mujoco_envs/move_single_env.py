from core.base_environment import *
from core.mujoco_environment import *
#Other imports
import numpy as np
import os
from os import path as osp
from overrides import overrides
from pyhelper_fns import vis_utils

MODULE_PATH = osp.dirname(osp.abspath(__file__))

def str2action(cmd):
  cmd = cmd.strip()
  if cmd == 'w':
    #up
    ctrl = [0, 0.1]
  elif cmd == 'a':
    #left
    ctrl = [-10, 0]
  elif cmd == 'd':
    #right
    ctrl = [10, 0]
  elif cmd == 's':
    #down
    ctrl = [0, 10]
  else:
    return None
  ctrl = 1000 * np.array(ctrl).reshape((2,))
  return ctrl 

class InitRandom(BaseInitializer):
  @overrides
  def sample_env_init(self):
    #Initialize all bodies within -0.25, 0.25
    pos  = 0.50 * np.random.rand(3,2) - 0.25
    keys = ['manipulator', 'target', 'object']  
    for i, k in enumerate(keys):
      self.simulator.set_body_pos2D(k, pos[i]) 


class ObsState(BaseObservation):
  @overrides
  def ndim(self):
    dim = {}
    dim['feat'] = 6
    return dim

  @overrides
  def observation(self):
    obs = {}
    obs['feat'] = np.zeros((6,))
    raise NotImplementedError
 
class ObsIm(BaseObservation):
  @overrides
  def ndim(self):
    dim = {}
    dim['im'] = (self.simulator.simParams['image_height'],
                 self.simulator.simParams['image_width'], 
                 3)
    return dim

  @overrides
  def observation(self):
    obs = {}
    obs['im'] =  self.simulator.get_image()
    return obs


class RewardSimple(BaseRewarder):
  #The radius around the goal in which reward is provided to the agent.
  @property
  def radius(self):
    return self.prms['radius'] if hasattr(self.prms, 'radius') else 0.1

  @overrides 
  def get(self):
    if self.simulator.dist_object_goal() < self.radius:
      return 1
    else:
      return 0 

class DiscreteActionFour(BaseDiscreteAction):
  @overrides
  def num_actions(self):
    return 4

  @overrides
  def process(self, action):
    assert len(action) == 1
    assert action[0] in [0, 1, 2, 3, 4]
    if action[0] == 0:
      #up
      ctrl = [0, 0.1]
    elif action[0] == 1:
      #left
      ctrl = [-0.1, 0]
    elif action[0] == 2:
      #right
      ctrl = [0.1, 0]
    elif action[0] == 3:
      #down
      ctrl = [0, -0.1]
    elif action[0] == 4:
      #no-op
      ctrl = [0, 0]
    else:
      raise Exception('Action %s not recognized' % action)
    ctrl = np.array(ctrl).reshape((2,))
    return ctrl 
 
  def minval(self):
    return 0

  def maxval(self):
    return 4


class ContinuousAction(BaseContinuousAction):
  @overrides
  def action_dim(self):
    return 2

  @overrides
  def process(self, action):
    return action  

  def minval(self):
    return -1.

  def maxval(self):
    return 1.

 
class MoveTeleportMujocoSimulator(BaseMujoco):
  def __init__(self, **kwargs):
    super(MoveTeleportMujocoSimulator, self).__init__(**kwargs)

  def _setup_renderer(self):
    if self.isRender:
      super(MoveTeleportMujocoSimulator, self)._setup_renderer()
      self.default_viewer.elevation = -90
      self.default_viewer.cam_distance = 0.8 
      self.default_viewer.cam.trackbodyid = -1
      self.render()

  def geom_names(self):
    return self.model.geom_names

  def _dist(self, x, y):
    dist = x - y
    dist = np.sqrt(np.sum(dist * dist))
    return dist

  def dist_manipulator_object(self):
    return self._dist(self.geom_xpos('manipulator_geom'), 
                      self.geom_xpos('object_geom'))

  def dist_object_goal(self):
    return self._dist(self.geom_xpos('object_geom'), 
                      self.geom_xpos('target_geom')) 
   

  
def get_environment(initName='InitRandom', obsName='ObsIm', rewName='RewardSimple',
                    actType='DiscreteActionFour', max_episode_length=100,
                    initPrms={}, obsPrms={}, rewPrms={}, actPrms={}, imSz=64):

  simParams = {}
  simParams['xmlfile'] = osp.join(MODULE_PATH, 'xmls/move_single.xml')
  simParams['image_width'] = imSz
  simParams['image_height'] = imSz
  sim     = MoveTeleportMujocoSimulator(simParams=simParams, isRender=True)
  initObj = globals()[initName](sim, initPrms)
  obsObj  = globals()[obsName](sim, obsPrms)
  rewObj  = globals()[rewName](sim, rewPrms)
  actObj  = globals()[actType](actPrms)
  env     = BaseEnvironment(sim, initObj, obsObj, rewObj, actObj,
                params={'max_episode_length': max_episode_length})
  return env
