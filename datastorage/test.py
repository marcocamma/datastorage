from __future__ import print_function
import numpy as np
from collections import OrderedDict
import datastorage
import time
import sys

def saveAndRead(obj,fname="/tmp/test.h5",link_copy=False):
  obj = datastorage.DataStorage(obj)
  t0 = time.time()
  obj.save(fname,link_copy=link_copy,raiseError=True)
  t1 = time.time()
  obj1 = datastorage.read(fname)
#  for k in obj:
#      print("  %s %s %s"%(k,str(obj[k])[:10],str(obj1[k])[:10]))
  t2 = time.time()
  return t2-t1,t1-t0

data = OrderedDict()

data[1] = dict(
  key1 = "this is a string",
  key2 = dict( key2_1 = "test", key2_2 = np.arange(1000) ),
  info = "saving dict ..."
)

# list 
data[2] = dict(
  key1 = "this is a string",
  key2 = [1,2,3],
  info = "saving list ..."
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

data[6] = dict(
  a= np.arange(100000),
  info = "this should take very little time ...(saving long array)"
)

data[7] = dict(
  a = [u'ciao',u'ciao1'],
  info = "list of unicode"
)

data[8] = dict(
  a = np.asarray([u'ciao',u'ciao1']),
  info = "array of unicode"
)

data[9] = dict(
  a = list(range(10000)),
  info = "long list 100000 elements"
)

data[10] = dict(
  a = list(range(10000)),
  b = None,
  info = "saving python None object"
)

  #d1 = dict( d2 = 3, d4=np.arange(10), d6=[1,2,3], d8 = [1,2,np.arange(10)] ),
data[11] = dict(
  d1 = dict( v1 = 3, v2=dict(vv1=np.arange(10)) ),
  info = 'nested dict'
)


data[12] = dict(
  d1 = dict( a = [1,2,3,np.arange(10)] ),
  info = 'complex list'
)

def _doTest(ext="h5"):
  keys = list(data.keys())
  keys.sort()
  for k in keys:
    v = data[k]
    tr,tw=saveAndRead(v,fname="/tmp/test_%02d.%s"%(k,ext))
    print("%2d read/write time %.4f,%.4f "%(k,tr,tw),end="")
    print(v["info"])
  a = np.random.random( (1000,1000) )
  tosave = dict( [ ("v%d"%i,a) for i in range(10)])
  tr,tw=saveAndRead( tosave,fname="/tmp/test_imgs_nolink.%s"%ext )
  print("   read/write time %.4f,%.4f Saving without links"%(tr,tw))
  tr,tw=saveAndRead( tosave,fname="/tmp/test_imgs_link.%s"%ext,link_copy=True )
  print("   read/write time %.4f,%.4f Saving with links"%(tr,tw))

def doTest( exts = ["h5","npy","npz"] ):
  print(datastorage)
  t0 = time.time()
  for ext in exts: 
    print("\n\nSaving in %s\n"%ext)
    _doTest(ext=ext)
  print("\n\n")
  print("Python version: %s"%sys.version)
  print("Time to complete all tests: %.1f"%(time.time()-t0))

if __name__ == "__main__": 
  doTest()
