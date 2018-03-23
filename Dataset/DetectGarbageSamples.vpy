import vapoursynth as vs
import Helpers
core = vs.get_core()
hi = core.raws.Source('Target.bin', 95, 95, src_fmt="GRAYS")
low = core.raws.Source('Guess.bin', 159, 159, src_fmt="GRAYS")
low = core.std.CropRel(low,32,32,32,32)
clp = core.std.MakeDiff(hi,low)
clp = Helpers.DetectGarbageSamples(clp, "log.txt")
clp.set_output()