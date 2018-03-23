import vapoursynth as vs
import os
import Helpers
core = vs.get_core()

Margin = (159 - 95) // 2

def exec(addr):
    clp = core.imwri.Read("lr2x_r/"+addr)
    #clp = core.fmtc.bitdepth(clp, bits=32, fulls=True, fulld=True)
    #clp = core.fmtc.transfer(clp, "srgb", "linear", fulls=True, fulld=True)
    #clp = core.std.ShufflePlanes(clp, 2, vs.GRAY)
    #clp = Helpers.sigmoid_inverse(clp)
    
    clp = Helpers.NonRingingResample(clp, clp.width*2, clp.height*2, "spline")
    clp = Helpers.ZeroCenterNormalization(clp)
    clp = Helpers.Mirror(clp, Margin, Margin+159, Margin, Margin+159)
    
    clp = core.imwri.Write(clp, "tif", "lr_r/"+addr+"%1d", 0, compression_type="None")
    return clp
    
    
clps = []
addrs = os.listdir("lr2x_r/")
for x in addrs:
    clps += [exec(x)]
clp = clps.pop(0)
for x in clps:
    clp = core.std.Splice([clp,x], True)
    
clp = core.std.CropAbs(clp, 1, 1, 0, 0)
clp.set_output()