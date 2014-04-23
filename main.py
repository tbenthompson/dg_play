import viscosaur as vc
import ctypes
import sys
import os
os.system('rm *.vtk')
mpi = ctypes.CDLL('libmpi.so.0', ctypes.RTLD_GLOBAL)
params = dict()
params['num_threads'] = 1
instance = vc.Vc(sys.argv, params)
dg = vc.DGMethod(3)
dg.run()
