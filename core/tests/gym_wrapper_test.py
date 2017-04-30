from .. import gym_wrapper
from .. import gym_wrapper_v2
from envs.mujoco_envs import move_single_env

def get_gymenv():
  env = move_single_env.get_environment(actType='ContinuousAction')
  env = gym_wrapper.GymWrapper(env)
  return env


def get_gymenv_v2():
  env = move_single_env.get_environment(actType='ContinuousAction')
  env = gym_wrapper_v2.GymWrapper(env)
  return env
