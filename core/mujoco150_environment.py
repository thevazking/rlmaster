import copy
from overrides import overrides
from .base_environment import *
#from pymj import cymj
from mujoco_py import cymj
from mujoco_py import const as MjCONST
from .my_mjviewer_utils import MjViewerExtended

AGENT_MUJOCO = {
    'substeps': 1,
    'camera_pos': np.array([2., 3., 2., 0., 0., 0.]),
     #This is for the free cameras. 
    'image_width': 320,
    'image_height': 320,
    'image_channels': 3,
}

def typeid_to_typename(typeid, objType='joint'):
  """
  Sample use: 
    ag.simple.jnt_type for instance returns 2 
    this function maps typeid '2' of objType 'joint' into the 
    name of the joint
  """
  hashList  = {'joint': 'JNT'}
  assert objType in hashList, 'objType %s not recognized' % objType
  hsh       = hashList[objType]
  names     = [k for k in MjCONST.__dict__.keys() if hsh in k]
  for n in names:
    if getattr(MjCONST, n) == typeid:
      return n
  raise Exception('Not Found')


class BaseMujoco150(BaseSimulator):
  """
  All communication between the algorithms and MuJoCo is done through
  this class.
  """
  def __init__(self, **kwargs):
    super(BaseMujoco150, self).__init__(**kwargs)
    if self.prms['xmlfile'] is not None:
      self._setup_world()
    self.camPrms = {}

  @property
  def defaultPrms(self):
    defPrms = super(BaseMujoco150, self).defaultPrms
    defPrms['xmlfile'] = None
    defPrms.update(AGENT_MUJOCO)
    return defPrms

  @property
  def model(self):
    return self.sim.model

  @property
  def data(self):
    return self.sim.data

  def set_xml(self, xmlfile):
    """
    Resets the environment with the xmlfile
    """
    self._prms['xmlfile'] = xmlfile
    self._setup_world()

  @overrides 
  def _setup_world(self):
    """
    Helper method for handling setup of the MuJoCo world.
    """
    fid    = open(self.simParams['xmlfile'], 'r')
    xmlStr = fid.read()
    self._model = cymj.load_model_from_xml(xmlStr)
    self.sim = cymj.MjSim(self._model)
    for nm in self.camera_names():
      self.camPrms[nm] = {}
      self.camPrms[nm]['width']  = self.prms['image_width']
      self.camPrms[nm]['height'] = self.prms['image_height']

  def camera_names(self):
    return self.sim.model.camera_names

  
  @overrides
  def render(self, cName=None):
    if cName is None:
      w, h = self.prms['image_width'], self.prms['image_height']
    else:
      w, h = self.camPrms[cName]['width'], self.camPrms[cName]['height']
    return self.sim.render(w, h, camera_name=cName)

  @overrides
  def get_image(self, cName=None):
    return self.render(cName=cName)
  
  @overrides
  def step(self, ctrl=None):
    #print (ctrl)
    if ctrl is not None:
      self.sim.data.ctrl[:] = copy.deepcopy(ctrl)
    for i in range(self.prms['actionRepeat']):
      self.sim.step()

  def save_image(self, fileName='default_im.png'):
    im = self.get_image()
    import matplotlib.pyplot as plt
    plt.imsave(fileName, im)
 

  def num_actuators(self):
    """
    Assumes each actuator is 1-D
    """
    return len(self.sim.model.actuator_trnid)


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
    bid  = self.sim.model.body_names.index(bodyName)
    bpos = self.sim.model.body_pos.copy()
    assert pos.shape == (3,) or pos.shape==(1,3)
    bpos[bid,:] = pos
    self.sim.model.body_pos = bpos
    self.sim.forward()
 
  def set_body_pos2D(self, bodyName, pos):
    bid  = self.sim.model.body_names.index(bodyName)
    bpos = self.sim.model.body_pos.copy()
    assert pos.shape == (2,) or pos.shape==(1,2)
    bpos[bid,0:2] = pos
    self.sim.model.body_pos = bpos
    self.sim.forward()

  def set_joint_pos(self, jntName, pos):
    adr      = self.sim.model.get_joint_qpos_addr(jntName)
    simState = self.sim.get_state()
    if type(adr) is tuple:
      st, en = adr
    else:
      st = adr
      en = st + 1
    simState.qpos[st:en] = copy.deepcopy(pos)
    self.sim.set_state(simState)
    self.sim.forward()

  def get_joint_pos(self, jntName):
    adr      = self.sim.model.get_joint_qpos_addr(jntName)
    #import IPython; IPython.embed()
    simState = self.sim.get_state()
    if type(adr) is tuple:
      st, en = adr
    else:
      st = adr
      #import IPython; IPython.embed()
      en = st + 1
    return copy.deepcopy(simState.qpos[st:en])



def simple_test(xmlfile=None):
  simParams = {}
  if xmlfile is None:
    xmlfile = '/work4/pulkitag-code/pkgs/mujoco/mjpro140/model/humanoid.xml'
  simParams['xmlfile'] = xmlfile
  ag = BaseMujoco150(simParams=simParams, isRender=True)
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
