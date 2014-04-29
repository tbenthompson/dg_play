import viscosaur as vc
import ctypes
import sys
import os
os.system('rm *.vtk')
mpi = ctypes.CDLL('libmpi.so.0', ctypes.RTLD_GLOBAL)
params = dict()
params['num_threads'] = 1
params['max_level'] = 4
params['min_level'] = 4
params['degree'] = 3
params['chi_refine'] = 0.25
params['chi_coarse'] = 0.10
params['epsilon_refine'] = 0.01
instance = vc.Vc(sys.argv, params)
dg = vc.DGMethod(params)
dg.run()
