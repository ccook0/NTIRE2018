import vapoursynth as vs
import Helpers
core = vs.get_core()
clp = core.raws.Source('Target.bin', 95, 95, src_fmt="GRAYS")
clp = Helpers.RemoveGarbageSamples(clp, "log.txt")
clp = Helpers.Augment(clp)
clp.set_output()