import vapoursynth as vs
import os
import Helpers
core = vs.get_core()

Folder = "Y2x/"

TopLeft = (159-95) // 2
BottomRight = TopLeft + 159

def exec(addr):
    clp = core.imwri.Read(Folder + addr)
    clp = Helpers.NonRingingResample(clp, clp.width * 2 , clp.height * 2, "spline")
    clp = Helpers.ZeroCenterNormalization(clp)
    
    clp = Helpers.Mirror(clp, TopLeft, BottomRight, TopLeft, BottomRight)
    clp = core.imwri.Write(clp, "tif", "Y4x_LR/"+addr+"%1d", 0, compression_type="None")
    return clp    
    
clps = []
addrs = os.listdir(Folder)
for x in addrs:
    clps += [exec(x)]
clp = clps.pop(0)
for x in clps:
    clp = core.std.Splice([clp,x], True)
    
clp = core.std.CropAbs(clp, 1, 1, 0, 0)
clp.set_output()