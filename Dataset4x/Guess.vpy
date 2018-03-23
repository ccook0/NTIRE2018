import vapoursynth as vs
import os
import Helpers
core = vs.get_core()

PatternSize = 159
Step = 87
Folder = "Y4x/"

def exec(addr):
    clp = core.imwri.Read(addr)
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