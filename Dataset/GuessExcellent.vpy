import vapoursynth as vs
import os
import Helpers
core = vs.get_core()

PatternSize = 159
Step = 87
Folder = "Excellent/"

def exec(addr):
    clp = core.imwri.Read(addr)
    clp = core.fmtc.bitdepth(clp, bits=32, fulls=True, fulld=True)
    clp = Helpers.CropByModN(clp, 2)
    clp = Helpers.sRGBToCIEYLinear(clp)
    clp = Helpers.sigmoid_inverse(clp)
    
    clp_low = Helpers.CatmullRomCubicDownsize(clp, 2)
    clp = Helpers.NonRingingResample(clp_low, clp.width, clp.height, "spline")
    clp = Helpers.ZeroCenterNormalization(clp)
    
    clp = Helpers.ExtractPatterns(clp, PatternSize, Step)
    return clp    
    
clps = []
addrs = os.listdir(Folder)
for x in addrs:
    clps += [exec(Folder + x)]
clp = clps.pop(0)
for x in clps:
    clp += x
clp.set_output()