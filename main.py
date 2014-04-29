import viscosaur as vc
import ctypes
import sys
import os
os.system('rm *.vtk')
mpi = ctypes.CDLL('libmpi.so.0', ctypes.RTLD_GLOBAL)
params = dict()
params['num_threads'] = 1
params['max_level'] = 6
params['min_level'] = 5
params['degree'] = 2
params['chi_refine'] = 100.0
params['chi_coarse'] = 20.0
params['epsilon_refine'] = 0.01
instance = vc.Vc(sys.argv, params)
dg = vc.DGMethod(params)
dg.run()
