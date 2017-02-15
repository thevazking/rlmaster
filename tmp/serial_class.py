import inspect
import subprocess
from multiprocessing import Pool

class A(object):
  def __init__(self):
    self.x = 1

  def add(self, y):
    self.x += y

def run_fn(a):
  print ('blah')
  a.add(10)
  return a.x

def run():
  pool = Pool(processes=6)
  a    = A()
  jobs = pool.map_async(run_fn, [a])
  res  = jobs.get()
  return res
  

