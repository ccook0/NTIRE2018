import vapoursynth as vs
import os
import Helpers
import functools
core = vs.get_core()

Margin = (159 - 95) // 2

def CancelCenterNormalization(ref, clp):
    stats = core.std.PlaneStats(ref)
    def SubtractMean(n, f, clp):
        mean = f.props["PlaneStatsAverage"]
        return core.std.Expr(clp, "x {mean} +".format(mean = mean))
    clp = core.std.FrameEval(clp, functools.partial(SubtractMean, clp = clp), prop_src = stats)
    return clp 

def exec(addr):
    clp = core.imwri.Read("hr_r/"+addr)
    ref = core.imwri.Read("ref_r/"+addr)
    clp = CancelCenterNormalization(ref, clp)
    clp = core.std.CropRel(clp, Margin, Margin+159, Margin, Margin+159)
    clp = core.imwri.Write(clp, "tif", "lr2x_r/"+addr+"%1d", 0, compression_type="None")
    return clp
    
    
clps = []
addrs = os.listdir("hr_r/")
for x in addrs:
    clps += [exec(x)]
clp = clps.pop(0)
for x in clps:
    clp = core.std.Splice([clp,x], True)
    
clp = core.std.CropAbs(clp, 1, 1, 0, 0)
clp.set_output()