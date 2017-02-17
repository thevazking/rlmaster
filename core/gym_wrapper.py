from core import base_environment

class GymWrapper(object):
  def __init__(self, env):
    self._env = env

  def _reset(self):
    self.env.reset()
    return self.env.observation()

  def reset(self):
    return self.reset()

  def _step(self, action):
    self.env.step(action)
    obs    = self.env.observation()
    reward = self.env.reward()
    done   = False
    return obs, reward, done, dict(reward=reward)

  def step(self):
    return self._step()

  def _get_obs(self):
    return self.env.observation()
 
  def viewer_setup(self):
    self.env._renderer_setup() 

  def render(self):
    return self.render()

  def _render(self):
    return self.env.render() 
  
