"""
Utilities for base_environment
"""
import numpy as np
from pyhelper_fns import vis_utils as vu
import matplotlib.pyplot as plt


def sample_random_action(dims, mode='randSample'):
  if mode == 'randSample':
    act = np.random.randn(dims,)
  elif mode == 'randUniform':
    act = 2 * np.random.randn(dims,) - 1
  else:
    act = 0.4 * np.zeros(dims,)
  return act


def visualize_random_exploration(env, mode='randSample', 
              numEpisodes=10, episodeLength=500):
  anim = vu.MyAnimation(None)
  if isinstance(env, BaseEnvironment):
    for e in range(10):
      _  = env.reset()
      for i in range(episodeLength):
        env.step(sample_random_action(env.simulator.num_actuators(), 
                                      mode=mode))
        im = env.simulator.get_image(cName='main')
        anim._display(im)
  else:
    raise Exception('Invalid argument type {0}'.format(type(env)))


def visualize_random_exploration_reward(env, mode='randUniform',
        numEpisodes=10, episodeLength=200, pause_val=0.005):
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

