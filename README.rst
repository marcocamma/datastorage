datastorage
--------

To use simply do::
   >>> from datastorage import DataStorage
   >>> data = DataStorage( a=(1,2,3),b="add",filename='store.npz' )

   >>> # data.a will be a dictionary
   >>> data = DataStorage( myinfo = dict( name= 'marco', surname= 'cammarata'),\
   >>>                     data   = np.arange(100) )

   >>>  # reads from file if it exists
   >>>  data = DataStorage( 'mysaveddata.npz' ) ;

   >>>  create empty storage (with default filename)
   >>>  data = DataStorage()
   >>>  data.mynewdata = np.ones(10)
