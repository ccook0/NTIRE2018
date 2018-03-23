import vapoursynth as vs
core = vs.get_core()
clp1 = core.raws.Source('TargetExcellent.bin', 95, 95, src_fmt="GRAYS")
clp2 = core.raws.Source('TargetFair.bin', 95, 95, src_fmt="GRAYS")
clp = clp1+clp2
clp.set_output()