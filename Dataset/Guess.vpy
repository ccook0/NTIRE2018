import vapoursynth as vs
core = vs.get_core()
clp1 = core.raws.Source('GuessExcellent.bin', 159, 159, src_fmt="GRAYS")
clp2 = core.raws.Source('GuessFair.bin', 159, 159, src_fmt="GRAYS")
clp = clp1+clp2
clp.set_output()