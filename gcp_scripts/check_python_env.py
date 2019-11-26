import getpass
import sys
import imp
import pandas
import keras
import numpy
import scipy
import elephas
import tensorflow

print('This job is running as "{}".'.format(getpass.getuser()))
print(sys.version_info)

print(imp.find_module('pandas'), pandas.__version__)
print(imp.find_module('keras'), keras.__version__)
print(imp.find_module('numpy'), numpy.__version__)
print(imp.find_module('scipy'), scipy.__version__)
print(imp.find_module('tensorflow'))
print(imp.find_module('elephas'))
