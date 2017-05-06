import copy
from overrides import overrides
from .base_environment import *
from pymj import cymj

AGENT_MUJOCO = {
    'substeps': 1,
    'camera_pos': np.array([2., 3., 2., 0., 0., 0.]),
    'image_width': 320,
    'image_height': 320,
    'image_channels': 3,
}


class BaseMujoco140(BaseSimulator):
  """
  All communication between the algorithms and MuJoCo is done through
  this class.
  """
  def __init__(self, **kwargs):
    super(BaseMujoco140, self).__init__(**kwargs)
    prms = copy.deepcopy(AGENT_MUJOCO)
    prms.update(self.simParams)
    self._simParams = prms
    assert 'xmlfile' in self.simParams, 'xmlfile must be provided'
    self._setup_world()

  #def __del__(self):
  #  self._small_viewer.finish()

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
    fid    = open(self.simParams['xmlfile'], 'r')
    xmlStr = fid.read()
    self.sim = cymj.MjSim(cymj.load_model_from_xml(xmlStr))
    
    #self._setup_renderer()
  def save_image(self):
    self.sim.reset()
    im = self.sim.render(100, 100)
    import matplotlib.pyplot as plt
    plt.imsave('blah_blah.png', im)

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
    gid = self.geom_name2id(geomName)
    return self.model.data.geom_xpos[gid]


  def set_body_pos(self, bodyName, pos):
    """
      Set the position of bodyName to pos
    """
    bid  = self.model.body_names.index(bodyName)
    bpos = self.model.body_pos.copy()
    assert pos.shape == (3,) or pos.shape==(1,3)
    bpos[bid,:] = pos
    self.model.body_pos = bpos
    self.model.forward()
 
  def set_body_pos2D(self, bodyName, pos):
    bid  = self.model.body_names.index(bodyName)
    bpos = self.model.body_pos.copy()
    assert pos.shape == (2,) or pos.shape==(1,2)
    bpos[bid,0:2] = pos
    self.model.body_pos = bpos
    self.model.forward()



def simple_test():
  simParams = {}
  DATA_DIR    = osp.join('/work4/pulkitag-code/pkgs/mujoco/mjpro140/model') 
  simParams['xmlfile'] = osp.join(DATA_DIR, 'humanoid.xml')
  ag = BaseMujoco140(simParams=simParams)
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
