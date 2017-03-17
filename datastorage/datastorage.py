""" npy/npz/hdf5 file based storage; 
    this modules adds the possibility to dump and load objects in files and
    a more convenient was of accessing the data via the .attributedict thanks
    to the DataStorage class """
import numpy as np
import os
import h5py
import collections
import logging
log = logging.getLogger(__name__)

_array_cache = {}


def unwrapArray(a, recursive=True, readH5pyDataset=True):
    """ This function takes an object (like a dictionary) and recursively
        unwraps it solving issues like:
          * the fact that many objects are packaged as 0d array
        This funciton has also some specific hack for handling h5py limits:
          * handle the None python object
          * numpy unicode ...
    """
    # is h5py dataset convert to array
    if isinstance(a, h5py.Dataset) and readH5pyDataset:
        a = a[...]
    if isinstance(a, h5py.Dataset) and a.shape == ():
        a = a[...]
    if isinstance(a, h5py.Group) and "IS_LIST_OF_ARRAYS" in a.attrs:
        items = list(a.keys())
        items.sort()
        a = np.asarray([a[item][...] for item in items])
    if isinstance(a, np.ndarray) and a.ndim == 0:
        a = a.item()
    if isinstance(a, np.ndarray) and a.dtype.char == "S":
        a = a.astype(str)
    if recursive:
        if "items" in dir(a):  # dict, h5py groups, npz file
            a = dict(a)  # convert to dict, otherwise can't asssign values
            for key, value in a.items():
                a[key] = unwrapArray(value)
        elif isinstance(a, (list, tuple)):
            a = [unwrapArray(element) for element in a]
        else:
            pass
    if isinstance(a, dict):
        a = DataStorage(a)
    # restore None that cannot be saved in h5py
    if isinstance(a, str) and a == "NONE_PYTHON_OBJECT":
        a = None
    # h5py can't save numpy unicode
    if isinstance(a, np.ndarray) and a.dtype.char == "S":
        a = a.astype(str)
    return a


def dictToH5Group(d, group, link_copy=True):
    """ helper function that transform (recursive) a dictionary into an
        hdf group by creating subgroups 
        link_copy = True, tries to save space in the hdf file by creating an internal link.
                    the current implementation uses memory though ...
    """
    global _array_cache
    for key, value in d.items():
        TOTRY = True
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
        if isinstance(value, dict):
            group.create_group(key)
            dictToH5Group(value, group[key], link_copy=link_copy)
        elif value is None:
            group[key] = "NONE_PYTHON_OBJECT"
        elif isinstance(value, np.ndarray):
            # take care of unicode (h5py can't handle numpy unicode arrays)
            if value.dtype.char == "U":
                value = np.asarray([vv.encode('ascii') for vv in value])
            # check if it is list of array
            elif isinstance(value, np.ndarray) and value.ndim == 1 and isinstance(value[0], np.ndarray):
                group.create_group(key)
                group[key].attrs["IS_LIST_OF_ARRAYS"] = True
                for index, array in enumerate(value):
                    dictToH5Group({"index%010d" % index: array},
                                  group[key], link_copy=link_copy)
                # don't even try to save as generic call group[key]=value
                TOTRY = False
            if link_copy:
                found_address = None
                for address, (file_handle, array) in _array_cache.items():
                    if np.array_equal(array, value) and group.file == file_handle:
                        log.info(
                            "Found array in cache, asked for %s/%s, found as %s" % (group.name, key, address))
                        found_address = address
                        break
                if found_address is not None:
                    value = group.file[found_address]
            try:
                if TOTRY:
                    group[key] = value
                    if link_copy:
                        log.info("Addind array %s to cache" %
                                 (group[key].name))
                        _array_cache[group[key].name] = (group.file, value)
            except Exception as e:
                log.warning("Can't save %s, error was %s" % (key, e))
        # try saving everything else that is not dict or array
        else:
            try:
                group[key] = value
            except Exception as e:
                log.error("Can't save %s, error was %s" % (key, e))


def dictToH5(h5, d, link_copy=False):
    """ Save a dictionary into an hdf5 file
        TODO: add capability of saving list of array
        h5py is not capable of handling dictionaries natively"""
    h5 = h5py.File(h5, mode="w")
#  group = h5.create_group("/")
    dictToH5Group(d, h5["/"], link_copy=link_copy)
    h5.close()


def h5ToDict(h5, readH5pyDataset=True):
    """ Read a hdf5 file into a dictionary """
    with h5py.File(h5, "r") as h:
        ret = unwrapArray(h, recursive=True, readH5pyDataset=readH5pyDataset)
    return ret


def npzToDict(npzFile):
    with np.load(npzFile) as npz:
        d = dict(npz)
    d = unwrapArray(d, recursive=True)
    return d


def npyToDict(npyFile):
    d = unwrapArray(np.load(npyFile).item(), recursive=True)
    return d


def dictToNpz(npzFile, d): np.savez(npzFile, **d)


def dictToNpy(npyFile, d): np.save(npyFile, d)


def objToDict(o, recursive=True):
    """ convert a DictWrap to a dictionary (useful for saving); it should work for other objects too 
        TODO: this function does not catch a list of DataStorage instances like
        objToDict( ( DataStorage(), DataStorage() ) )
        is not converted !!
    """
    if "items" not in dir(o):
        return o
    d = dict()
    for k, v in o.items():
        try:
            d[k] = objToDict(v)
        except Exception as e:
            log.info("In objToDict, could not convert key %s to dict, error was" %
                     (k, e))
            d[k] = v
    return d


def read(fname):
    extension = os.path.splitext(fname)[1]
    log.info("Reading storage file %s" % fname)
    if extension == ".npz":
        return DataStorage(npzToDict(fname))
    elif extension == ".npy":
        return DataStorage(npyToDict(fname))
    elif extension == ".h5":
        return DataStorage(h5ToDict(fname))
    else:
        raise ValueError(
            "Extension must be h5, npy or npz, it was %s" % extension)


def save(fname, d, link_copy=True):
    """ link_copy is used by hdf5 saving only, it allows to creat link of identical arrays (saving space) """
    # make sure the object is dict (recursively) this allows reading it
    # without the DataStorage module
    d = objToDict(d, recursive=True)
    d['filename'] = fname
    extension = os.path.splitext(fname)[1]
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


class DataStorage(dict):
    """ Storage for dict like object.
        recursive : bool
           recursively convert dict-like objects to DataStorage
        It can save data to file (format npy,npz or h5)

        To initialize it:

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

    def __init__(self, *args, filename='data_storage.npz', recursive=True, **kwargs):
        #    self.filename = kwargs.pop('filename',"data_storage.npz")
        self.filename = filename
        self._recursive = recursive
        # interpret kwargs as dict if there are
        if len(kwargs) != 0:
            fileOrDict = dict(kwargs)
        elif len(kwargs) == 0 and len(args) > 0:
            fileOrDict = args[0]
        else:
            fileOrDict = dict()

        d = dict()  # data dictionary
        if isinstance(fileOrDict, dict):
            d = fileOrDict
        elif isinstance(fileOrDict, str):
            if os.path.isfile(fileOrDict):
                d = read(fileOrDict)
            else:
                self.filename = fileOrDict
                d = dict()
        else:
            raise ValueError("Invalid DataStorage definition")

        if recursive:
            for k in d.keys():
                if not isinstance(d[k], DataStorage) and isinstance(d[k], dict):
                    d[k] = DataStorage(d[k])

        # allow accessing with .data, .delays, etc.
        for k, v in d.items():
            setattr(self, k, v)

        # allow accessing as proper dict
        self.update(**dict(d))

    def __setitem__(self, key, value):
        # print("__setitem__")
        setattr(self, key, value)
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        """ allows to add fields with data.test=4 """
        # check if attr exists is essential (or it fails when defining an
        # instance)
        if hasattr(self, "_recursive") and self._recursive and \
                isinstance(value, (dict, collections.OrderedDict)):
              value = DataStorage(value)
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __delitem__(self, key):
        delattr(self, key)
        super().__delitem__(key)

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
                all([isinstance(v, np.ndarray) for v in obj]):
                value_str = "list of arrays, shapes " + \
                    ",".join([str(v.shape) for v in obj[:5]]) + " ..."
            elif isinstance(obj, np.ndarray):
                value_str = "array, size %s, type %s" % (
                    "x".join(map(str, obj.shape)), obj.dtype)
            elif isinstance(obj, DataStorage):
                value_str = str(obj)[:50]
            elif isinstance(obj, (str, DataStorage)):
                value_str = obj[:50]
            elif self[k] is None:
                value_str = "None"
            else:
                value_str = str(self[k])
            if len(str(obj)) > 50:
                value_str += " ..."
            s.append(fmt % (k, value_str))
        return "\n".join(s)

    def keys(self):
        keys = list(super().keys())
        keys = [k for k in keys if k != 'filename']
        keys = [k for k in keys if k[0] != '_']
        return keys

    def save(self, fname=None, link_copy=False):
        """ link_copy: only works in hfd5 format
            save space by creating link when identical arrays are found,
            it slows down the saving (3 or 4 folds) but saves A LOT of space
            when saving different dataset together (since it does not duplicate
            internal pyfai matrices
        """
        if fname is None:
            fname = self.filename
        assert fname is not None
        save(fname, self, link_copy=link_copy)