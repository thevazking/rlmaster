from rlmaster.core.base_environment import *
import numpy as np
from overrides import overrides
from pyhelper_fns import vis_utils

class DefaultParams(object):
  def sim(self):
    params  = {'nDistractor': 2,
               'nMovable'   : 0,
                'imSz'      : 42}
    return params

DEF_PARAM = DefaultParams()


def cmd2action(dirmag):
  """
  Args:
    cmd, mag = dirmag
    cmd: which direction to move
    mag: the magnitude of motion
  """
  if type(cmd) is str:
    cmd = cmd.strip()
  if cmd == 'w' or cmd == 0:
    #up
    ctrl = [0, 1]
  elif cmd == 'a' or cmd == 1:
    #left
    ctrl = [-1, 0]
  elif cmd == 'd' or cmd == 2:
    #right
    ctrl = [1, 0]
  elif cmd == 's' or cmd == 3:
    #down
    ctrl = [0, -1]
  else:
    raise Exception('Unexpected action')
  ctrl = np.array(ctrl).reshape((1,2))
  
  return ctrl


class DiscreteActionFour(BaseDiscreteAction):
  @overrides
  def num_actions(self):
    return 4

  def minval(self):
    return 0

  def maxval(self):
    return 3

  @overrides
  def process(self, action):
    ctrl = cmd2action()    

class Simulator(BaseSimulator):
  def __init__(self, **kwargs):
    """
      simParams (in kwargs):
        nDistractor: number of distractor objects
        nMovable   : number of movable objects
    """
    super(Simulator, self).__init__(defaults=DEF_PARAM.sim(),
                                    **kwargs)
    self.pos = {}
    self.pos['distractor'] = np.zeros((self.simParams['nDistractor'], 2))
    self.pos['movable']    = np.zeros((self.simParams['nMovable'], 2))
    self.pos['ball']       = np.zeros((1,2))

  @overrides
  def step(self, ctrl):
    """
    Args:
      ctrl: assumed to be a tuple of (direction, mag)
    """ 
    direction, mag = ctrl
    if direction == 0
     

    

