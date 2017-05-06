import numpy as np
import datastorage
from datastorage import DataStorage as ds


## create storage ##

# empty
data = ds()

# from a dict
data = ds( dict( key1 = 'value1' ) )

# with keywords arguments
data = ds( key1 = 3 )

## adding stuff ... ##

# as if it is a dict
data['key2'] = 1234

# this is even nicer ... ;)
data.key3 = 34

# addidng another key (by default it will converted to a DataStorage instance if possible)
data.key4 = dict( key4_1 = 3, key4_2 = "ciao")

print("is key4 a datastorage ?",isinstance(data.key4,ds))

# can also handle lists/tuples ...
data.key5 = [1,2,3]

# but this will not work...(or rather will not be saved)
# data.key5 = [1,np.arange(10)]

# convert object to Datastorage
class MM:
    def __init__(self):
        self.f = sum
ds(MM())


## Save and read ##

nested=dict( data1 = np.random.random( (100,100) ), info = "this is a test" )

# adding a datastorage within another one
data1 = ds( key1 = np.arange(100), key2 = nested, rawdata=data)

# save in npz file format
data1.save("/tmp/datastorage_example.npz")

# save in hdf5 files
data1.save("/tmp/datastorage_example.h5")

# and reading it again
data = datastorage.read("/tmp/datastorage_example.h5")
