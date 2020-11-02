from __future__ import print_function
""" npy/npz/hdf5 file based storage; 
    this modules adds the possibility to dump and load objects in files and
    a more convenient was of accessing the data via the .attributedict thanks
    to the DataStorage class """
import numpy as np
import os
import sys
import math
import h5py
import collections
import logging
import pathlib
log = logging.getLogger(__name__)


# make sure dictionaries are ordered (somehow it does not work !)
#dict = collections.OrderedDict

_array_cache = dict()


def unwrapArray(a, recursive=True, readH5pyDataset=True):
    """ This function takes an object (like a dictionary) and recursively
        unwraps it solving issues like:
          * the fact that many objects are packaged as 0d array
        This funciton has also some specific hack for handling h5py limits:
          * handle the None python object
          * numpy unicode ...
    """
    try:

        ### take care of hdf5 groups 
        if isinstance(a,h5py.Group):
            # take care of special flags first
            if isinstance(a, h5py.Group) and ( ("IS_LIST" in a.attrs) or ("IS_LIST_OF_ARRAYS" in a.attrs) ):
                items = list(a.keys())
                items.sort()
                a = [unwrapArray(a[item],readH5pyDataset=readH5pyDataset) for item in items]


        ### take care of hdf5 datasets
        elif isinstance(a,h5py.Dataset):

            # read if asked so or if dummy array
            # WARNING: a.value and a[...] do not return the
            # same thing... 
            # a[...] returns ndarray if a is a string
            # a.value returns a str(py3) or unicode(py2)
            if readH5pyDataset or a.shape == (): a = a[()] #.value#[...]


        # special None flag
        # not array needed for FutureWarning: elementwise comparison failed; ...
        if not isinstance(a,np.ndarray) and (a == "NONE_PYTHON_OBJECT" or a == b"NONE_PYTHON_OBJECT"): a = None
 
        # clean up non-hdf5 specific
        if isinstance(a, np.ndarray) and a.ndim == 0:
            a = a.item()

        # convert to str (for example h5py can't save numpy unicode)
        if isinstance(a, np.ndarray) and a.dtype.char == "S":
            a = a.astype(str)

        if recursive:
            if "items" in dir(a):  # dict, h5py groups, npz file
                a = dict(a)  # convert to dict, otherwise can't asssign values
                for key, value in a.items():
                    a[key] = unwrapArray(value,readH5pyDataset=readH5pyDataset)
            elif isinstance(a, (list, tuple)):
                a = [unwrapArray(element,readH5pyDataset=readH5pyDataset) 
                    for element in a]
            else:
                pass

    except Exception as e:
        log.warning("Could not handle %s, error was: %s"%(a,str(e)))
    return a




def _find_link(value,group,key):
    global _array_cache
    name = "%s/%s" % (group.name,key)
    name = name.replace("//","/")
    if not isinstance(value,np.ndarray): return None
    found_address = None
    for address, array in _array_cache.items():
        if np.array_equal(array, value):
            log.info("Found array in cache, asked for %s, found as %s" % (name,address))
            found_address = address
            break
    if found_address is not None:
        value = group.file[found_address]
    else:
        log.info("Adding array %s to cache" % name)
        _array_cache[name] = value
#    print(list(_array_cache.keys()))
    return value



def dictToH5Group(d, group, link_copy=True):
    """ helper function that transform (recursive) a dictionary into an
        hdf group by creating subgroups 
        link_copy = True, tries to save space in the hdf file by creating an internal link.
                    the current implementation uses memory though ...
    """
    for key in d.keys():
        value = d[key]
        log.debug("saving",key,"in",group)
        # hope for the best (i.e. h5py can handle that)
        try:
            if link_copy and isinstance(value,np.ndarray):
              value=_find_link(value,group,key)
            else:
              if isinstance(value,h5py.Dataset): value = value[:]; # hdf5 dataset have to be read first ...
            group[key] = value
        except (TypeError,ValueError) as e:
            log.debug("For %s, h5py could not handle the saving on its own, trying to convert it, error was %s"%(key,e))
            if isinstance(value,dict) or hasattr(value,"__dict__"):
                if key not in group: group.create_group(key)
                try:
                    value = dictToH5Group(value,group[key],link_copy=link_copy)
                # objects have __dict__ but can be coverted to dict like only 
                # by DataStorage (and not by dict)
                except:
                    value = dictToH5Group(DataStorage(value),group[key],link_copy=link_copy)
            # take care of unicode (h5py can't handle numpy unicode arrays)
            elif isinstance(value,np.ndarray) and value.dtype.char == "U":
                value = np.asarray([vv.encode('ascii') for vv in value])
                group[key] = value
            elif isinstance(value, collections.Iterable):
                if key not in group: group.create_group(key)
                group[key].attrs["IS_LIST"] = True
                fmt = "index%%0%dd" % math.ceil(np.log10(len(value)))
                for index, array in enumerate(value):
                    dictToH5Group({fmt % index: array},
                              group[key], link_copy=link_copy)
            elif value is None:
                group[key] = "NONE_PYTHON_OBJECT"
            else:
                log.warn("Could not convert %s into an object that can be saved"%key)


def dictToH5(h5, d, link_copy=False):
    """ Save a dictionary into an hdf5 file
        h5py is not capable of handling dictionaries natively"""
    global _array_cache
    _array_cache = dict()
    h5 = h5py.File(h5, mode="w")
    dictToH5Group(d, h5["/"], link_copy=link_copy)
    h5.close()
    _array_cache = dict(); # clean up memory ...


def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    h = h5py.File(h5, "r")
    ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    if readH5pyDataset: h.close()
    return ret


def npzToDict(npzFile):
    with np.load(npzFile, allow_pickle=True) as npz:
        d = dict(npz)
    d = unwrapArray(d, recursive=True)
    return d


def npyToDict(npyFile):
    d = unwrapArray(np.load(str(npyFile), allow_pickle=True).item(), recursive=True)
    return d


def dictToNpz(npzFile, d): np.savez(npzFile, **d)


def dictToNpy(npyFile, d): np.save(npyFile, d)

def _toDict(datastorage_obj,recursive=True):
    """ this is the recursive part of the toDict (otherwise it fails when converting to DataStorage """
    if "items" not in dir(datastorage_obj): return datastorage_obj
    d = dict()
    for k, v in datastorage_obj.items():
        try:
            d[k] = _toDict(v)
        except Exception as e:
            log.info("In toDict, could not convert key %s to dict, error was %s" %
                     (k, e))
            d[k] = v
    return d
 

def toDict(datastorage_obj, recursive=True):
    """ convert a DataStorage object to a dictionary (useful for saving); it should work for other objects too 
    """
    # if not a DataStorage, convert to it first
    if "items" not in dir(datastorage_obj): datastorage_obj = DataStorage(datastorage_obj)
    return _toDict(datastorage_obj)

def read(fname,raiseError=True,readH5pyDataset=True):
    fname = pathlib.Path(fname)
    err_msg = "File " + str(fname) + " does not exist"
    if not fname.is_file():
        if raiseError:
            raise ValueError(err_msg)
        else:
            log.error(err_msg)
            return None
    extension = fname.suffix
    log.info("Reading storage file %s" % fname)
    if extension == ".npz":
        return DataStorage(npzToDict(fname))
    elif extension == ".npy":
        return DataStorage(npyToDict(fname))
    elif extension == ".h5":
        return DataStorage(h5ToDict(fname,readH5pyDataset=readH5pyDataset))
    else:
        try:
            return DataStorage(h5ToDict(fname,readH5pyDataset=readH5pyDataset))
        except Exception as e:
            err_msg = "Could not read " + str(fname) + " as hdf5 file, error was: %s"%e
            log.error(err_msg)
            if raiseError:
                raise ValueError(err_msg)
            else:
                return None


def save(fname, d, link_copy=True,raiseError=False):
    """ link_copy is used by hdf5 saving only, it allows to creat link of identical arrays (saving space) """
    # make sure the object is dict (recursively) this allows reading it
    # without the DataStorage module
    fname = pathlib.Path(fname)
    d = toDict(d, recursive=True)
    d['filename'] = str(fname)
    extension = fname.suffix
    log.info("Saving storage file %s" % fname)
    try:
        if extension == ".npz":
            return dictToNpz(fname, d)
        elif extension == ".h5":
            return dictToH5(fname, d, link_copy=link_copy)
        elif extension == ".npy":
            return dictToNpy(fname, d)
        else:
            raise ValueError(
                "Extension must be h5, npy or npz, it was %s" % extension)
    except Exception as e:
        log.exception("Could not save %s" % fname)
        if raiseError: raise  e


class DataStorage(dict):
    """ Storage for dict like object. It also tries to convert general
        objects to instances by using __dict__

        It can save data to file (format npy,npz or h5)

        Parameters
        ----------
        filename : str
           default filename to use for saving
        recursive : bool
           recursively convert dict-like objects to DataStorage

        Examples
        --------
          data = DataStorage( a=(1,2,3),b="add",filename='store.npz' )

          # recursively by default
          # data.a will be a DataStorage instance
          data = DataStorage( a=dict( b = 1)) );

          # data.a will be a dictionary
          data = DataStorage( a=dict( b = 1),recursive=False )

          # reads from file if it exists
          data = DataStorage( 'mysaveddata.npz' ) ;

          DOES NOT READ FROM FILE (even if it exists)!!
          data = DataStorage( filename = 'mysaveddata.npz' ); 

          create empty storage (with default filename)
          data = DataStorage()
    """

    def __init__(self, *args, **kwargs):
        self.filename = kwargs.pop('filename',"data_storage.npz")
        self._recursive = kwargs.pop('recursive',True)

        # interpret kwargs as dict if there are
        if len(kwargs) != 0:
            input_data = dict(kwargs)
        elif len(kwargs) == 0 and len(args) > 0:
            input_data = args[0]
        else:
            input_data = dict()

        d = dict()  # data dictionary
        if isinstance(input_data, dict):
            d = input_data
        elif isinstance(input_data, str):
            if os.path.isfile(input_data):
                d = read(input_data)
            else:
                self.filename = input_data
                d = dict()
        elif isinstance(input_data,np.ndarray) and input_data.dtype.names is not None:
            for name in input_data.dtype.names:
                d[name] = input_data[name]
        else:
            try:
                d = input_data.__dict__
            except AttributeError:
                log.error("Could not interpret input as object to package")
                raise ValueError("Invalid DataStorage definition")

        if self._recursive:
            for k in d.keys():
                if not isinstance(d[k], DataStorage) and isinstance(d[k], dict):
                    d[k] = DataStorage(d[k])

        # allow accessing with .data, .delays, etc.
        for k, v in d.items():
            setattr(self, k, v)

        # allow accessing as proper dict
        self.update(**dict(d))

    def __setitem__(self, key, value):
        """ method to add a key via obj["key"] = value """
        #print("__setitem__")
        setattr(self, key, value)

    def __setattr__(self, key, value):
        """ method to add a key via obj.key = value """
        # check if attr exists is essential (or it fails when defining an
        # instance)
        #print("self.__setattr__")
        if hasattr(self, "_recursive") and self._recursive and \
                isinstance(value, (dict, collections.OrderedDict)):
              value = DataStorage(value)
        super(DataStorage,self).__setitem__(key, value)
        super(DataStorage,self).__setattr__(key, value)

    def __delitem__(self, key):
        delattr(self, key)
        super(DataStorage,self).__delitem__(key)

    def __str__(self):
        keys = list(self.keys())
        keys.sort()
        return "DataStorage obj containing: %s" % ",".join(keys)

    def __repr__(self):
        keys = list(self.keys())
        keys.sort()
        if len(keys) == 0:
            return "Empty DataStorage"
        nchars = max(map(len, keys))
        fmt = "%%%ds %%s" % (nchars)
        s = ["DataStorage obj containing (sorted): ", ]
        for k in keys:
            if k[0] == "_":
                continue
            obj = self[k]
            if ((isinstance(obj, np.ndarray) and obj.ndim == 1) or \
                isinstance(obj, (list, tuple))) and \
                all((isinstance(v, np.ndarray) for v in obj)):
                value_str = "list of arrays, shapes " + \
                    ",".join([str(v.shape) for v in obj[:5]]) + " ..."
            elif isinstance(obj, np.ndarray):
                value_str = "array, size %s, type %s" % (
                    "x".join(map(str, obj.shape)), obj.dtype)
            elif isinstance(obj, DataStorage):
                value_str = str(obj)[:50]
            elif isinstance(obj, str):
                value_str = obj[:50].replace('\r\n','\n').replace("\n"," ")
            elif self[k] is None:
                value_str = "None"
            else: 
                value_str = str(self[k]).replace('\r\n','\n').replace("\n"," ")
            if len(str(obj)) > 50:
                value_str += " ..."
            s.append(fmt % (k, value_str))
        return "\n".join(s)

#    def __getitem__(self,x):
#      if x[0] != "_": return x

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
      keys = self.keys()
      for k in keys:
          yield k,self[k]

    def update(self,dictionary=None,**kwargs):
      if dictionary is None: dictionary=kwargs
      for (key,value) in dictionary.items(): self.__setitem__(key,value)

    def toDict(self): return toDict(self)

    def keys(self):
        keys = list(super(DataStorage,self).keys())
        keys = [k for k in keys if k != 'filename']
        keys = [k for k in keys if k[0] != '_']
        return keys

    def save(self, fname=None, link_copy=False,raiseError=False):
        """ link_copy: only works in hfd5 format
            save space by creating link when identical arrays are found,
            it may slows down the saving (3 or 4 folds) but saves space
            when saving different dataset together (since it does not duplicate
            arrays)
        """
        if fname is None:
            fname = self.filename
        assert fname is not None
        save(fname, self, link_copy=link_copy,raiseError=raiseError)


def unwrap(list_of_datastorages):
    """ 
    convert list of data storages in one datastorage with array elements
    useful in conjuction with list comprehension

    def f(x):
        return DataStorage(x=x,x2=x**2,x3=x**3)

    res = [f(i) for i in range(10)]
    res = unwrap(res)
    """
    retout=DataStorage()
    for key in list_of_datastorages[0].keys():
        retout[key] = np.asarray([r[key] for r in list_of_datastorages])
    return retout
    useful
