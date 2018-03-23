import numpy

def LoadRawBinaryGrayscaleSequence(Address, Width, Height, Length, PixelType='float32'):
    Sequence = numpy.memmap(Address, dtype=PixelType, mode='r', shape=(Length, Height, Width, 1))
    return Sequence

def ExportRawBinaryGrayscaleSequence(Data, Address, PixelType='float32'):
    Sequence = numpy.memmap(Address, dtype=PixelType, mode='w+', shape=Data.shape)
    Sequence[:] = numpy.array(Data, dtype=PixelType)[:]
    del Sequence
    
def LoadRawBinaryGrayscaleSequenceAdvanced(Address, Width, Height, Length):
    Sequence = numpy.memmap(Address, dtype='float32', mode='r', shape=(Length, Height, Width, 1))
    return Sequence
