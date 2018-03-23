import os
addrs = os.listdir(".")
for x in addrs:
    y = os.path.splitext(x)
    if y[1] == ".tif0":
       os.rename(x, y[0]+".tif")