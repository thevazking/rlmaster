"""
Utilities for base_environment
"""
import abc
import numpy as np
from pyhelper_fns import vis_utils as vu
import matplotlib.pyplot as plt
from rlmaster.core.base_environment import BaseEnvironment

def visualize_resets(env):
  """
  Visualize the environment after a reset has been 
  performed
  """
  anim = vu.MyAnimation(None)
  for i in range(100):
    _  = env.reset()
    im = env.simulator.get_image(cName='main')
    anim._display(im)
    ip = input()
    if ip == 'q':
      return


def sample_random_action(dims, mode='randSample'):
  if mode == 'randSample':
    act = np.random.randn(dims,)
  elif mode == 'randUniform':
    act = 2 * np.random.randn(dims,) - 1
  else:
    act = 0.4 * np.zeros(dims,)
  return act


class BaseEnvVis(object):
  __metaclass__ = abc.ABCMeta
  def __init__(self, env, mode='randSample', numEpisodes=10, 
               interactiveFn=None):
    """
    Args:
      env : instance of BaseEnvironment
      mode: the mode in which actions should be sampled
            this is overwritten by interactiveFn if it is not None
      numEpisodes  : number of episodes to run for.
      interactiveFn: the function used for interactive visualization 
    """
    assert isinstance(env, BaseEnvironment),\
              'Invalid argument type {0}'.format(type(env))
    self.env  = env
    self.mode = mode
    self.numEpisodes   = numEpisodes
    self.episodeLength = self.env.params['max_episode_length']
    self.interactiveFn = interactiveFn

  def get_image(self):
    return self.env.simulator.get_image(cName='main')

  def step(self):
    if self.interactiveFn is not None:
      self.env.interactive_step(self.interactiveFn)
    else:
      self.env.step(sample_random_action(self.env.simulator.num_actuators(), 
                                      mode=self.mode))
        
  def vis_resets(self):
    anim = vu.MyAnimation(None)
    for i in range(100):
      self.env.reset()
      anim._display(self.get_image())
      ip = input()
      if ip == 'q':
        return

  def vis_exploration(self):
    anim = vu.MyAnimation(None)
    for e in range(self.numEpisodes):
      env.reset()
      for i in range(self.episodeLength):
        self.step()
        anim._display(self.get_image())
    del anim
    plt.close('all')

  def vis_exploration_touch(self, pauseVal=0.005):
    nTouch = self.env.touch_ndim() 
    names  = list(self.env.simulator.model.sensor_names)
    animation = vu.MyAnimationMulti(None, numPlots= 1 + nTouch,
                          isIm=[True] + nTouch*[False],
                          axTitles=['im'] + names)
    plt.ion()
    for e in range(self.numEpisodes):
      self.env.reset()
      plt.show()
      hpt = []
      for n in range(nTouch):
        hpt.append([])
      for i in range(self.episodeLength):
        self.step()
        touch  = self.env.simulator.data.sensordata
        for n in range(nTouch):
          hpt[n].append(touch[n])
        pltTuple = []
        for n in range(nTouch):
          pltTuple.append((range(len(hpt[n])), hpt[n]))
        animation._display([self.get_image()] + pltTuple)
        plt.pause(pauseVal)


def visualize_random_exploration_reward(env, mode='randUniform',
        numEpisodes=10, episodeLength=200, pause_val=0.005):
  """
  Visualize the agent alongwith the reward during random exploration
  """
  plt.ion()
  anim = vu.MyAnimationMulti(None, subPlotShape=(1,2),
                            isIm=[True, False])
  while True:
    env.reset()
    rew = []
    for i in range(episodeLength):
      env.step(sample_random_action(env.simulator.num_actuators(), 
                                    mode=mode))
      rew.append(env.reward())
      if np.mod(i,20)==1:
          anim._display([env.simulator.get_image(cName='main'), 
                              (range(len(rew)),rew)])
      plt.pause(pause_val) 
      

def save_random_exploration_video(env, mode='randSample', 
              numEpisodes=10, episodeLength=500):
  vidName = 'random_exploration.mp4'
  vid     = vu.VideoMaker()
  if isinstance(env, BaseEnvironment):
    for e in range(numEpisodes):
      _  = env.reset()
      for i in range(episodeLength):
        env.step(sample_random_action(env.simulator.num_actuators(), 
                                      mode=mode))
        im = env.simulator.get_image(cName='main')
        vid.save_frame(im)
  else:
    raise Exception('Invalid argument type {0}'.format(type(env)))
  vid.compile_video()
