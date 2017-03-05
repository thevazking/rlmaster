from .. import gym_wrapper
from envs.mujoco_envs import move_single_env

def get_gymenv():
  env = move_single_env.get_environment(actType='ContinuousAction')
  env = gym_wrapper.GymWrapper(env)
  return env
