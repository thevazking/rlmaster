import copy
from overrides import overrides
from .base_environment import *
from mujoco_py_wrapper import mujoco_py as mjcpy

AGENT_MUJOCO = {
    'substeps': 1,
    'camera_pos': np.array([2., 3., 2., 0., 0., 0.]),
    'image_width': 640,
    'image_height': 480,
    'image_channels': 3,
}

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

  @overrides 
  def _setup_world(self):
    """
    Helper method for handling setup of the MuJoCo world.
    """
    self._model = mjcpy.MjModel(self.simParams['xmlfile'])
    #self._joint_idx = list(range(self._model['nq']))
    #self._vel_idx = [i + self._model['nq'] for i in self._joint_idx]
    self._setup_renderer()

  @overrides
  def _setup_renderer(self):
      if self.isRender:
        cam_pos = self.simParams['camera_pos']
        gofast = True
        self._small_viewer = mjcpy.MjViewer(visible=True,
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


  def interactive(self):
    while True:
      isValid = False
      ip      = raw_input()
      if ip == 'q':
        break
      else:
        cmds = ip.split(',')
        if not len(cmds) == 2:
          continue
        else:
          ctrl = np.array([float(cmds[0]), float(cmds[1])])
      self.step_by_n(5, ctrl)
      self.render()


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
