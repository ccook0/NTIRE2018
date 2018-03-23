from imageio import imread, imwrite
import numpy
import keras
import os
import tensorflow as tf
from keras.utils import multi_gpu_model

with tf.device('/cpu:0'):
     Model = keras.models.load_model('Model-002.h5')
Model = multi_gpu_model(Model, gpus=2)

def exec(addr):
    img = imread("Y4x_LR/"+addr)
    Size=159
    Margin=(159-95)//2
    Step=Size - 2 * Margin
    Height = img.shape[0]
    Width = img.shape[1]
    PatternList = []
    for i in range(0, Height - Size + 1, Step):
        for j in range(0, Width - Size + 1, Step):
            PatternList += [img[i + 0:i + Size, j + 0:j + Size]]
    PatternCount = len(PatternList)
    Array = numpy.zeros((PatternCount,) + PatternList[0].shape + (1,))
    for i in range(PatternCount):
        Array[i,:,:,0] = PatternList[i]    
    ReconstructedPatterns = Model.predict(Array,batch_size=48,verbose=True)
    Size=95
    Canvas = numpy.zeros((Height, Width),dtype='float32')
    HorizontalPatternCount = (Width - 2 * Margin) // Size
    for i in range(0, Height - Size - 2 * Margin + 1, Size):
        for j in range(0, Width - Size - 2 * Margin + 1, Size):
            CorrespondingIndex = (i * HorizontalPatternCount + j) // Size
            Canvas[i + Margin:i + Margin + Size, j + Margin:j + Margin + Size] = ReconstructedPatterns[CorrespondingIndex,:,:,0]
    imwrite("Y4x_HR/"+addr, Canvas, format="TIFF")
    return

addrs = os.listdir("Y4x_LR/")
for x in addrs:
    exec(x)