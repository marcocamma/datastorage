import numpy as np
from collections import OrderedDict
import datastorage

def saveAndRead(obj,fname="/tmp/test.h5"):
  obj = datastorage.DataStorage(obj)
  obj.save(fname)
  obj1 = datastorage.read(fname)

data = OrderedDict()

data[1] = dict(
  key1 = "this is a string",
  key2 = dict( key2_1 = "test", key2_2 = np.arange(1000) )
)

# list 
data[2] = dict(
  key1 = "this is a string",
  key2 = [1,2,3]
) 

# list 
data[3] = dict(
  info = "saving list of arrays (same shape)",
  key1 = "this is a string",
  key2 = [np.arange(10),np.arange(10)*2,np.arange(10)*3]
) 

# list 
data[4] = dict(
  info = "saving list of arrays (different shape)",
  key1 = "this is a string",
  key2 = [np.arange(10),np.arange(20)*2,np.arange(30)*3]
) 

data[5] = dict(
  info = "saving list of stuff",
  key1 = "this is a string",
  key2 = [np.arange(10),dict(key2_1=3,key2_2=np.arange(20)*2)]
) 


def doTest():
  for v in data.values(): saveAndRead(v)
  print("All passed")
