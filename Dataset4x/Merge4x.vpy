import vapoursynth as vs
import os
import Helpers
import functools
core = vs.get_core()

Margin = (159 - 95) // 2

def exec(addr):
    clp = core.imwri.Read("Y4x_HR/"+addr)
    clp = core.std.CropRel(clp, Margin, Margin+159, Margin, Margin+159)
    clp = core.imwri.Write(clp, "tif", "Y4x/"+addr+"%1d", 0, compression_type="None")
    return clp
    
    
clps = []
addrs = os.listdir("Y4x_HR/")
for x in addrs:
    clps += [exec(x)]
clp = clps.pop(0)
for x in clps:
    clp = core.std.Splice([clp,x], True)
    
clp = core.std.CropAbs(clp, 1, 1, 0, 0)
clp.set_output()