from simple_agent_mujoco import *
import numpy as np
from pyhelper_fns import vis_utils

class MoveTeleportSimulator(BaseSimulator):
  def __init__(self, **kwargs):
    self._pos = {}
    self._pos['manipulator'] = np.zeros((2,))
    self._pos['object']      = np.zeros((2,))
    self._pos['goal']        = np.zeros((2,))
    #Maximum and minimum locations of objects
    self._range_min = -1
    self._range_max = 1
    #Manipulate radius
    self._manipulate_radius = 0.2
    #Image size
    self._imSz = 64 
    self._im = np.zeros((self._imSz, self._imSz, 3), dtype=np.uint8)      

  def action_ndim(self):
    return 2

  def _dist(self, x, y):
    dist = x - y
    dist = np.sqrt(np.sum(dist * dist))
    return dist

  def dist_manipulator_object(self):
    return self._dist(self._pos['manipulator'], self._pos['object'])

  def dist_object_goal(self):
    return self._dist(self._pos['object'], self._pos['goal']) 
   
  def _clip(self, val):
    val = np.clip(val, self._range_min, self._range_max)

  def step(self, ctrl):
    self._pos['manipulator'] += ctrl.reshape((2,))
    self._clip(self._pos['manipulator']) 
    if self.dist_manipulator_object() < self._manipulate_radius:
      self._pos['object'] = self._pos['manipulator'].copy()  
 
  def _get_bin(self, rng, coords):
    x = np.where(rng < coords[0])[0][-1]
    y = np.where(rng < coords[0])[0][-1]        
    return x, y

  def _plot_object(self, coords, color='r'):
    x, y = coords
    mnx, mxx  = max(0, x - 2), min(self._imSz, x + 2)
    mny, mxy  = max(0, y - 2), min(self._imSz, y + 2)
    if color == 'r':
      self._im[mny:mxy, mnx:mxx, 0] = 255
    elif color == 'g':
      self._im[mny:mxy, mnx:mxx, 1] = 255
    else:
      self._im[mny:mxy, mnx:mxx, 2] = 255
      
    
  def get_image(self):
    imSz = self._imSz
    rng = np.linspace(self._range_min, self._range_max, imSz)
    m_x, m_y = self._get_bin(rng, self._pos['manipulator'])
    o_x, o_y = self._get_bin(rng, self._pos['object'])
    self._im = np.zeros((imSz, imSz, 3), dtype=np.uint8)      
    self._plot_object(self._pos['object'], 'r')
    self._plot_object(self._pos['goal'], 'g')
    self._plot_object(self._pos['manipulator'], 'b')
     

class MoveTeleportObsState(BaseObservation):
  def ndim(self):
    dim = {}
    dim['feat'] = 6

  def observation(self):
    obs = np.zeros((6,))
    for i, k in enumerate(self.simulator._pos.keys()):
      obs[2*i, 2*i + 2] = self.simulator._pos[k].copy()
    return obs


class MoveTeleportObsState(BaseObservation):
  def ndim(self):
    dim = {}
    dim['feat'] = 6

  def observation(self):
    obs = np.zeros((6,))
    for i, k in enumerate(self.simulator._pos.keys()):
      obs[2*i, 2*i + 2] = self.simulator._pos[k].copy()
    return obs
  
 
class MoveTeleportObsIm(BaseObservation):
  def ndim(self):
    dim = {}
    dim['im'] = (self.simulator._imSz, self.simulator._imSz, 3)

  def observation(self):
    return self.simulator.get_image()
