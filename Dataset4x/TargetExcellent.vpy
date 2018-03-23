import vapoursynth as vs
import os
import Helpers
core = vs.get_core()

Folder = "Excellent/"

def exec(addr):
    clp = core.imwri.Read(Folder + addr)
    clp = core.fmtc.bitdepth(clp, bits=32, fulls=True, fulld=True)
    clp = Helpers.CropByModN(clp, 4)
    clp = Helpers.sRGBToCIEYLinear(clp)
    clp = Helpers.sigmoid_inverse(clp)
    clp = Helpers.ZeroCenterNormalization(clp)

    clp = core.imwri.Write(clp, "tif", "YHR/"+addr+"%1d", 0, compression_type="None")
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