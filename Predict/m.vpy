import vapoursynth as vs
import os
import Helpers
import functools
core = vs.get_core()



def exec(addr):
    r = core.imwri.Read("planeR/"+addr)
    r = Helpers.sigmoid_direct(r)
    r = core.fmtc.transfer(r, "linear", "srgb", fulls=True, fulld=True)

    g = core.imwri.Read("planeG/"+addr)
    g = Helpers.sigmoid_direct(g)
    g = core.fmtc.transfer(g, "linear", "srgb", fulls=True, fulld=True)


    b = core.imwri.Read("planeB/"+addr)
    b = Helpers.sigmoid_direct(b)
    b = core.fmtc.transfer(b, "linear", "srgb", fulls=True, fulld=True)

    clp = core.std.ShufflePlanes([r,g,b], 0, vs.RGB)
    clp = core.fmtc.bitdepth(clp, bits=8, fulls=True, fulld=True, dmode=6)
    clp = core.imwri.Write(clp, "png", "8x/"+addr+"%1d", 0)
    return clp
    
    
clps = []
addrs = os.listdir("planeR/")
for x in addrs:
    clps += [exec(x)]
clp = clps.pop(0)
for x in clps:
    clp = core.std.Splice([clp,x], True)
    
clp = core.std.CropAbs(clp, 1, 1, 0, 0)
clp.set_output()