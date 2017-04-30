import copy
from overrides import overrides
from .base_environment import *
#import mujoco_py as mjcpy
#from mujoco_py import mjviewer
from mujoco_py_wrapper import mujoco_py as mjcpy
print (mjcpy.__file__)
from mujoco_py_wrapper.mujoco_py import mjviewer

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


class MujocoCanvas(object):
  def __init__(self, height=200, width=200, ncam=1):
    """
    Args:
      ncam: number of cameras
    """
    self._width = width
    self._height = height
    self._ncam = ncam
    self.osmesa = osmesa_wrapper.PyOSMesaContext(self._width,
                                                 self._height)
    self.osmesa.MakeCurrent(self._width, self._height)
    self.scene = core.MjvScene()  # abstract visualization objects
    self.cam = []
    for n in range(self._ncam):
      self.cam.append(core.MjvCamera())  # camera control, window-specific
    self.vopt = core.MjvOption()  # visualization options, window-specific
    self.con = core.MjrContext()  # custom OpenGL context
    self.sel_pos = np.zeros(3, np.double)

    #A maximum of 1000 objects will need to be made.
    mjlib.mjv_makeScene(self.scene.ptr, 1000)
    mjlib.mjv_defaultOption(self.vopt.ptr)
    mjlib.mjr_defaultContext(self.con.ptr)
    for n in range(self._ncam):
      mjlib.mjv_defaultCamera(self.cam[n].ptr)
    #The font-scaling is 150. Not sure what this means.
    #mjlib.mjr_makeContext(None, self.con.ptr, 150)
    self.cam[0].camid = -1
    if self._ncam == 2:
      self.cam[1].camid = 1
      self.cam[1].type_ = enums.mjtCamera.mjCAMERA_TRACKING
      self.cam[1].trackbodyid = 3
      self.cam[1].elevation = -90
  @property
  def height(self):
    return self._height

  @property
  def width(self):
    return self._width

  def setup(self, model, data, autoscale=False):
    # re-create custom context
    mjlib.mjr_makeContext(model.ptr, self.con.ptr, 150)

    #direct rendeing ro off-screen buffer
    mjlib.mjr_setBuffer(enums.mjtFramebuffer.mjFB_OFFSCREEN, self.con.ptr)

    if autoscale:
      # center and scale view
      self.autoscale(model, data)

    #Create the rectangle and buffer for storing the rendering
    self.rect = []
    self.buf  = []
    for n in range(self._ncam):
      self.rect.append(types.MJRRECT(0, 0, self._width, self._height))
      self.buf.append(np.zeros((self._height, self._width, 3), dtype=np.uint8))
      
  def autoscale(self, model, data, cam_id=0):
    """Set the camera in an appropriate position."""
    #with util.SetFlags(self.cam[cam_id].lookat, writeable=True) as lookat:
    #  lookat[:] = model.stat.center[:]
    #for i in range(3):
    #  self.cam[cam_id].lookat[i] = model.stat.center[i]
    self.cam[cam_id].distance = 1.5 * model.stat.extent
    self.cam[cam_id].camid = -1
    self.cam[cam_id].trackbodyid = -1
    #mjlib.mjv_updateCameraPose(self.cam.ptr, self._width * 1.0 / self._height)

  def set_camera_pose(self, azimuth=None, elevation=None, cam_id=0):
    """
      Args:
        azimuth: in degrees
        elevation: in degrees
        cam_id: which camera to change
    """
    if azimuth is not None:
      self.cam[cam_id].azimuth = azimuth
    if elevation is not None:
      self.cam[cam_id].elevation = elevation
 
  def set_camera_distance(self, distance, cam_id=0):
    self.cam[cam_id].distance = distance

  def render(self, model, data):
    """Render the current state of the model."""
    #Make the scene
    mjlib.mjv_updateScene(model.ptr, data.ptr, self.vopt.ptr, None,
                          self.cam[0].ptr, enums.mjtCatBit.mjCAT_ALL, self.scene.ptr)
    for n in range(self._ncam):
      #Use camera n
      self.snapshot_to_camera(model, data, self.cam[n], self.rect[n], self.buf[n])

  def snapshot_to_camera(self, model, data, camera, rect, buf):
    mjlib.mjv_updateCamera(model.ptr, data.ptr, camera.ptr, self.scene.ptr)
    mjlib.mjr_render(rect, self.scene.ptr, self.con.ptr)
    mjlib.mjr_readPixels(buf, None, rect, self.con.ptr)


  def get_camera_image_coords(self, cam_id=0):
    """
      Returns the coordinate of the image captured by the camera 
      in the mujoco coordinate system
    """
    cam  = self.cam[cam_id]
    #Assumes the camera is looking downards on the plan.
    assert np.abs(cam.elevation) == 90
    assert self.height==self.width, 'Only accurate when height and width are same'
    dist  = cam.distance
    #The viewport of the camera is pi/4, so half-angle is pi/8
    side =  dist * math.tan(np.pi/8)
    x_min = cam.lookat[0] - side
    x_max = cam.lookat[0] + side
    y_min = cam.lookat[1] - side
    y_max = cam.lookat[1] + side
    return x_min, y_min, x_max, y_max

  def to_pixel_coords(self, pt, cam_id=0, as_float=False):
    x_min, y_min, x_max, y_max = self.get_camera_image_coords(cam_id)
    x_bins = np.linspace(x_min, x_max, self.width + 1)
    y_bins = np.linspace(y_min, y_max, self.height + 1)
    if pt[0] < x_min or pt[0] > x_max:
      return None, None
    if pt[1] < y_min or pt[1] > y_max:
      return None, None
    if as_float:
      x = (pt[0]- x_min)/(x_max - x_min) * self.width
      #To account for y-axis inversion
      y = (pt[1] - y_min)/(y_max - y_min) * self.height
      y = self.height - y
    else:
      x = np.where(pt[0] >= x_bins)[0][-1]
      #To account for y-axis inversion
      y = np.where(-pt[1] >= y_bins)[0][-1]
    return x, y

  def square_around_pt_in_pixels(self, pt, sz, cam_id=0):
    """
      pt: point in mujoco coordinate frame
      sz: size/2 in mujoco coordinate
    """
    assert self.height==self.width, 'Only accurate when height and width are same'
    x, y = pt
    side = self.cam[cam_id].distance * math.tan(np.pi/8)
    x1   = max(-side, x - sz)
    y1   = max(-side, y - sz)
    x2   = min(side, x + sz)
    y2   = min(side, y + sz)
    x1, y1 = self.to_pixel_coords([x1, y1], cam_id=cam_id)
    x2, y2 = self.to_pixel_coords([x2, y2], cam_id=cam_id)
    return x1, y2, x2, y1

  def get_image(self, model, data):
    self.render(model, data)
    res = ()
    for n in range(self._ncam):
      res = res + (self.buf[n][::-1],)
    return res

  def draw_image(self, model, data, ax=None):
    if ax is None:
      fig = plt.figure()
      ax  = fig.add_subplot(111)
    ax.imshow(self.get_image(model, data))


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
