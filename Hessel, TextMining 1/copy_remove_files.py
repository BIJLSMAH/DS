# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:13:30 2020

@author: Laptop
"""


#!/usr/bin/python

import os
import fnmatch
from shutil import copyfile

# Import your data to a Pandas.DataFrame
mapTM = r"D:\Foto's\@eaDir"
hoofdmap = "%s" % (mapTM)
os.chdir(hoofdmap)


vn = 1
for root, dir, files in os.walk("."):
        print(root)
        print("")
        for items in fnmatch.filter(files, "*"):
                print("..." + items)
                vn += 1
                print("van " + root + "\\" + items + " naar " + hoofdmap + '\\' + str(vn) + "_" + str.replace(root, ".\\", ""))
#                copyfile(root + "\\" + items, hoofdmap + '\\' + str(vn) + "_" + str.replace(root, ".\\", ""))
        print("")
        
for root, dir, files in os.walk("."):
    print(root)
    print("")
    for items in fnmatch.filter(files, "*"):
        if (os.path.getsize(root + '\\' + items) < 100000):
            print(os.path.getsize(root + '\\' + items))    
            os.remove(root + '\\' + items)
        print("..." + items)
    print("")