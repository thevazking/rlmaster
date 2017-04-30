import numpy as np
from rlmaster.core import base_environment
from pyhelper_fns import vis_utils

class DmNavAction(base_environment.BaseDiscreteAction):
  def num_actions(self):
    return 13

  def minval(self):
    raise Exception('Not Implemented')

  def maxval(self):
    raise Exception('Not implemented')

  def process(self, action):
    """
    Converts action from the agent into environment
    Args:
      action: from the agent
    """
    action = int(action)
    ctrl   = np.zeros((7,))
    if action == 0:
      #no-op
      pass
    if action == 1:
      #look-left large
      ctrl[0] = -40.
    elif action == 2:
      #look-left medium
      ctrl[0] = -20.
    elif action == 3:
      #look-left small
      ctrl[0] = -10.
    elif action == 4:
      #look-right small
      ctrl[0] = 10.
    elif action == 5:
      #look-right medium
      ctrl[0] = 20.
    elif action == 6:
      #look-right medium
      ctrl[0] = 40.
    elif action == 7:
      #look down
      ctrl[1] = -20.
    elif action == 8:
      #look up
      ctrl[1] = 20.
    elif action == 9:
      #strafe left
      ctrl[2] = -1.
    elif action == 10:
      #strafe right
      ctrl[2] = 1.
    elif action == 11:
      #move back
      ctrl[3] = -1.
    elif action == 12:
      #move forward
      ctrl[3] = 1
    return ctrl.astype(np.intc)      


class DmNavActionSimple(base_environment.BaseDiscreteAction):
  def num_actions(self):
    return 5

  def minval(self):
    raise Exception('Not Implemented')

  def maxval(self):
    raise Exception('Not implemented')

  def process(self, action):
    """
    Converts action from the agent into environment
    Args:
      action: from the agent
    """
    action = int(action)
    ctrl   = np.zeros((7,))
    if action == 0:
      #no-op
      pass
    elif action == 1:
      #look-left
      ctrl[0] = -50.
    elif action == 2:
      #look-right
      ctrl[0] = 50.
    elif action == 3:
      #move forward
      ctrl[3] = 1.
    elif action == 4:
      #move back
      ctrl[3] = -1
    return ctrl.astype(np.intc)      



class Dm2GymWrapper(object):
  def __init__(self, env, actProc):
    """
    Args:
      env      : instance of dm_lab
                 eg: env = deepmind_lab.Lab('nav_maze_random_goal_01', 
                                    ['RGB_INTERLACED'], 
                                    config={'width':'48', 'height':'48'})
      actProc  : a processor that converts action
                 into the desired format
    """
    self._env = env
    self._actProc = actProc

  @property
  def env(self):
    return self._env

  @property
  def actProc(self):
    return self._actProc

  def num_actions(self):
    return self.actProc.num_actions()
 
  def observation(self):
    """
    Only 1 type of observation supported for now
    """ 
    return self.env.observations()['RGB_INTERLACED']

  def get_rgb_obs_size(self):
    """
    Returns the size of the RGB observation
    """
    for o in self.env.observation_spec():
      if o['name'] == 'RGB_INTERLACED':
        return o['shape'][0], o['shape'][1]
    raise Exception('RGB Observations not found')

  def reset(self):
    self.env.reset()
 
  def step(self, action):
    #Process the action
    act    = self.actProc.process(action)
    print (act)
    reward = self.env.step(act)
    obs    = self.observation()
    done   = self.env.is_running()
    return obs, reward, done, dict(reward=reward) 

  def viewer_setup(self):
    h, w = self.get_rgb_obs_size()
    self._canvas = vis_utils.MyAnimation(None, height=h, width=w)

  def render(self):
    self._canvas._display(self.observation()) 

  def interactive(self):
    self.viewer_setup()
    while True:
      ip = raw_input()
      if ip == 'q':
        return
      elif int(ip) <= 13:
        self.step(int(ip))
        self.render()
      else:
        print ('Unrecognized command')
