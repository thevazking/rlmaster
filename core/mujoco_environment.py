import copy
from overrides import overrides
from .base_environment import *
import mujoco_py as mjcpy
from mujoco_py import mjviewer

AGENT_MUJOCO = {
    'substeps': 1,
    'camera_pos': np.array([2., 3., 2., 0., 0., 0.]),
    'image_width': 320,
    'image_height': 320,
    'image_channels': 3,
}

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

  
class BaseMujoco(BaseSimulator):
  """
  All communication between the algorithms and MuJoCo is done through
  this class.
  """
  def __init__(self, **kwargs):
    super(BaseMujoco, self).__init__(**kwargs)
    prms = copy.deepcopy(AGENT_MUJOCO)
    prms.update(self.simParams)
    self._simParams = prms
    assert self.simParams.has_key('xmlfile'), 'xmlfile must be provided'
    self._setup_world()


  def __del__(self):
    self._small_viewer.finish()


  @property
  def model(self):
    return self._model

  @property
  def default_viewer(self):
    return self._small_viewer

  @overrides 
  def _setup_world(self):
    """
    Helper method for handling setup of the MuJoCo world.
    """
    self._model = mjcpy.MjModel(self.simParams['xmlfile'])
    #self._geoms = 
    #self._joint_idx = list(range(self._model['nq']))
    #self._vel_idx = [i + self._model['nq'] for i in self._joint_idx]
    self._setup_renderer()

  @overrides
  def _setup_renderer(self):
    if self.isRender:
      cam_pos = self.simParams['camera_pos']
      gofast = True
      self._small_viewer = MjViewerExtended(visible=True,
                                          init_width=self.simParams['image_width'],
                                          init_height=self.simParams['image_height'],
                                          go_fast=gofast)
      self._small_viewer.start()
      self._small_viewer.set_model(self._model)
      self.render()

  @overrides
  def render(self):
    print ('I am here')
    self._small_viewer.render()
    self._small_viewer.loop_once()

  @overrides
  def step(self, ctrl=None):
    if ctrl is not None:
      self._model.data.ctrl = copy.deepcopy(ctrl)
    self.model.step()

  @overrides
  def get_image(self):
      img_string, width, height = self._small_viewer.get_image()
      img = np.fromstring(img_string, dtype='uint8').reshape(
            (height, width, self.simParams['image_channels']))[::-1, :, :]
      return img

  def geom_name2id(self, geomName):
    return self.model.geom_names.index(geomName)

  def geom_xpos(self, geomName):
    """
    position of a geom in global-coordinate frame
    Args:
      geomName: name of the geom
    """
    gid = self.geom_name2id()
    return self.model.data.geom_xpos[gid]


def simple_test():
  hyperparams = {}
  #DATA_DIR   = osp.join(os.getenv('HOME'), 'code', 'gps', 'mjc_models') 
  DATA_DIR    = osp.join('/work4/pulkitag-code/pkgs/','gps', 'mjc_models') 
  hyperparams['filename'] = osp.join(DATA_DIR, 'particle2d.xml')
  check_utils.exists(hyperparams['filename']) 
  ag = BaseAgent(hyperparams)
  return ag 
  ctrl = np.array(([1., 1.]))
  print (ag.model.data.xpos)
  ag.render()
  im1  = ag.get_image()
  ag.step_by_n(1000, ctrl)
  print (ag.model.data.xpos)
  ag.render()
  im2  = ag.get_image()
  vu.plot_pairs(im1, im2) 
  return ag
