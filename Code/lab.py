from __lib__ import *

root = '/home/dtan/Downloads/'
for (root,dirs,files) in os.walk(root,topdown=True):
        print (root)
        print (dirs)
        print (files)
        print ('--------------------------------')
