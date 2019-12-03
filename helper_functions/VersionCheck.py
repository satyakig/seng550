import getpass
import sys
import imp

print("This job is running as {}".format(getpass.getuser()))
print(sys.executable, sys.version_info)
for package in sys.argv[1:]:
    print(imp.find_module(package))
