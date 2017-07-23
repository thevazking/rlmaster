import copy
from overrides import overrides
from  mujoco_py import mjviewer

class MjViewerExtended(mjviewer.MjViewer):
  @property
  def cam_lookat(self):
    return self.cam.lookat
  
  @cam_lookat.setter
  def cam_lookat(self, lookat):
    lookat  = copy.deepcopy(lookat)
    for i in range(3):
      self.cam.lookat[i] = lookat[i]
   
  def cam_lookat_x(self, x):
    self.cam.lookat[0] = x

  def cam_lookat_y(self, y):
    self.cam.lookat[1] = y

  def cam_lookat_z(self, z):
    self.cam.lookat[2] = z

  @property
  def cam_distance(self):
    return self.cam.distance

  @cam_distance.setter
  def cam_distance(self, d):
    self.cam.distance = d
    
  @property
  def azimuth(self):
    return self.cam.azimuth

  @azimuth.setter
  def azimuth(self, az):
    self.cam.azimuth = az

  @property
  def elevation(self):
    return self.cam.elevation

  @elevation.setter
  def elevation(self, el):
    self.cam.elevation = el


