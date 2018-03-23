from keras import models, layers, optimizers
from keras.utils import multi_gpu_model
import tensorflow as tf
import functools

PatternSize = 159

def VGGConv(Input, FilterCount, Tag, Activation=True):
    Conv = layers.Conv2D(FilterCount, (3, 3), activation=None, padding='valid', name='Conv' + Tag)(Input)
    if Activation:
       Conv = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + Tag)(Conv)
    return Conv

def GetBlockA(Input, BlockOfHigherOrder=None, Crop=None):
    Crop = 0 if BlockOfHigherOrder is None and Crop is None else Crop
    Conv = VGGConv(Input, 64, 'AE1')
    ConvRaw = VGGConv(Conv, 64, 'AE2', False)
    Conv = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'AE2')(ConvRaw)
    LowResFeatures = layers.convolutional.Cropping2D(2)(Input)
    LowResFeatures = layers.concatenate([LowResFeatures, Conv])
    Map = VGGConv(LowResFeatures, 64, 'AD2')
    MapRaw = VGGConv(Map, 64, 'AD1', False)
    ConvRaw = layers.convolutional.Cropping2D(2)(ConvRaw)
    MapRaw = layers.add([ConvRaw, MapRaw])    
    Map = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'AD1')(MapRaw)
    if BlockOfHigherOrder is not None:
        LowResFeatures = layers.convolutional.Cropping2D(2)(LowResFeatures)
        HiResFeatures = Map
        Map = BlockOfHigherOrder(HiResFeatures, LowResFeatures, MapRaw)
    else:
        Map = layers.convolutional.Cropping2D(Crop)(Map)
    MergeLayer = layers.convolutional.Cropping2D(Crop + 4)(Input)
    Ensemble = layers.Conv2D(1, (1, 1), activation=None, use_bias=False, padding='valid', name='EnsembleA')(Map)
    Ensemble = layers.add([MergeLayer, Ensemble])
    return Ensemble

def GetBlockB(HiResFeatures, LowResFeatures, MergeLayer, BlockOfHigherOrder=None, Crop=None):
    Crop = 0 if BlockOfHigherOrder is None and Crop is None else Crop
    Conv = VGGConv(HiResFeatures, 128, 'BE1')
    ConvRaw = VGGConv(Conv, 128, 'BE2', False)
    Conv = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'BE2')(ConvRaw)
    LowResFeatures = layers.convolutional.Cropping2D(2)(LowResFeatures)
    LowResFeatures = layers.concatenate([LowResFeatures, Conv])
    Map = VGGConv(LowResFeatures, 128, 'BD2')
    MapRaw = VGGConv(Map, 128, 'BD1', False)
    ConvRaw = layers.convolutional.Cropping2D(2)(ConvRaw)
    MapRaw = layers.add([ConvRaw, MapRaw])    
    Map = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'BD1')(MapRaw)
    if BlockOfHigherOrder is not None:
        LowResFeatures = layers.convolutional.Cropping2D(2)(LowResFeatures)
        HiResFeatures = layers.convolutional.Cropping2D(4)(HiResFeatures)
        HiResFeatures = layers.concatenate([HiResFeatures, Map])
        Map = BlockOfHigherOrder(HiResFeatures, LowResFeatures, MapRaw)
    else:
        Map = layers.convolutional.Cropping2D(Crop)(Map)
    MergeLayer = layers.convolutional.Cropping2D(Crop + 4)(MergeLayer)
    Ensemble = layers.Conv2D(64, (1, 1), activation=None, padding='valid', name='EnsembleB')(Map)
    Ensemble = layers.add([MergeLayer, Ensemble])
    Ensemble = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'EnsembleB')(Ensemble)
    return Ensemble

def GetBlockC(HiResFeatures, LowResFeatures, MergeLayer, BlockOfHigherOrder=None, Crop=None):
    Crop = 0 if BlockOfHigherOrder is None and Crop is None else Crop
    Conv = VGGConv(HiResFeatures, 256, 'CE1')
    Conv = VGGConv(Conv, 256, 'CE2')
    Conv = VGGConv(Conv, 256, 'CE3')
    ConvRaw = VGGConv(Conv, 256, 'CE4', False)
    Conv = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'CE4')(ConvRaw)
    LowResFeatures = layers.convolutional.Cropping2D(4)(LowResFeatures)
    LowResFeatures = layers.concatenate([LowResFeatures, Conv])
    Map = VGGConv(LowResFeatures, 256, 'CD4')
    Map = VGGConv(Map, 256, 'CD3')
    Map = VGGConv(Map, 256, 'CD2')
    MapRaw = VGGConv(Map, 256, 'CD1', False)
    ConvRaw = layers.convolutional.Cropping2D(4)(ConvRaw)
    MapRaw = layers.add([ConvRaw, MapRaw])  
    Map = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'CD1')(MapRaw)
    if BlockOfHigherOrder is not None:
        LowResFeatures = layers.convolutional.Cropping2D(4)(LowResFeatures)
        HiResFeatures = layers.convolutional.Cropping2D(8)(HiResFeatures)
        HiResFeatures = layers.concatenate([HiResFeatures, Map])
        Map = BlockOfHigherOrder(HiResFeatures, LowResFeatures, MapRaw)
    else:
        Map = layers.convolutional.Cropping2D(Crop)(Map)
    MergeLayer = layers.convolutional.Cropping2D(Crop + 8)(MergeLayer)
    Ensemble = layers.Conv2D(128, (1, 1), activation=None, padding='valid', name='EnsembleC')(Map)
    Ensemble = layers.add([MergeLayer, Ensemble])
    Ensemble = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'EnsembleC')(Ensemble)
    return Ensemble

def GetBlockD(HiResFeatures, LowResFeatures, MergeLayer, BlockOfHigherOrder=None, Crop=None):
    Crop = 0 if BlockOfHigherOrder is None and Crop is None else Crop
    Conv = VGGConv(HiResFeatures, 512, 'DE1')
    Conv = VGGConv(Conv, 512, 'DE2')
    Conv = VGGConv(Conv, 512, 'DE3')
    ConvRaw = VGGConv(Conv, 512, 'DE4', False)
    Conv = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'DE4')(ConvRaw)
    LowResFeatures = layers.convolutional.Cropping2D(4)(LowResFeatures)
    LowResFeatures = layers.concatenate([LowResFeatures, Conv])
    Map = VGGConv(LowResFeatures, 512, 'DD4')
    Map = VGGConv(Map, 512, 'DD3')
    Map = VGGConv(Map, 512, 'DD2')
    MapRaw = VGGConv(Map, 512, 'DD1', False)
    ConvRaw = layers.convolutional.Cropping2D(4)(ConvRaw)
    MapRaw = layers.add([ConvRaw, MapRaw])   
    Map = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'DD1')(MapRaw)
    if BlockOfHigherOrder is not None:
        LowResFeatures = layers.convolutional.Cropping2D(4)(LowResFeatures)
        HiResFeatures = layers.convolutional.Cropping2D(8)(HiResFeatures)
        HiResFeatures = layers.concatenate([HiResFeatures, Map])
        Map = BlockOfHigherOrder(HiResFeatures, LowResFeatures, MapRaw)
    else:
        Map = layers.convolutional.Cropping2D(Crop)(Map)
    MergeLayer = layers.convolutional.Cropping2D(Crop + 8)(MergeLayer)
    Ensemble = layers.Conv2D(256, (1, 1), activation=None, padding='valid', name='EnsembleD')(Map)
    Ensemble = layers.add([MergeLayer, Ensemble])
    Ensemble = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'EnsembleD')(Ensemble)
    return Ensemble

def GetBlockE(HiResFeatures, LowResFeatures, MergeLayer):
    Conv = VGGConv(HiResFeatures, 512, 'EE1')
    Conv = VGGConv(Conv, 512, 'EE2')
    Conv = VGGConv(Conv, 512, 'EE3')
    ConvRaw = VGGConv(Conv, 512, 'EE4', False)
    Conv = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'EE4')(ConvRaw)
    LowResFeatures = layers.convolutional.Cropping2D(4)(LowResFeatures)
    LowResFeatures = layers.concatenate([LowResFeatures, Conv])
    Map = VGGConv(LowResFeatures, 512, 'ED4')
    Map = VGGConv(Map, 512, 'ED3')
    Map = VGGConv(Map, 512, 'ED2')
    MapRaw = VGGConv(Map, 512, 'ED1', False)
    ConvRaw = layers.convolutional.Cropping2D(4)(ConvRaw)
    MapRaw = layers.add([ConvRaw, MapRaw])
    Map = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'ED1')(MapRaw)
    MergeLayer = layers.convolutional.Cropping2D(8)(MergeLayer)
    Ensemble = layers.Conv2D(512, (1, 1), activation=None, padding='valid', name='EnsembleE')(Map)
    Ensemble = layers.add([MergeLayer, Ensemble])
    Ensemble = layers.advanced_activations.PReLU(shared_axes=[1, 2], name='PReLu' + 'EnsembleE')(Ensemble)
    return Ensemble

def GetModel(lr, gpus):
    with tf.device('/cpu:0'):
         HiFreqGuess = layers.Input(shape=(PatternSize, PatternSize, 1), name='HiFreqGuess')
         GenerateBlockE = GetBlockE
         GenerateBlockD = functools.partial(GetBlockD, BlockOfHigherOrder=GenerateBlockE, Crop=8)    
         GenerateBlockC = functools.partial(GetBlockC, BlockOfHigherOrder=GenerateBlockD, Crop=16) 
         GenerateBlockB = functools.partial(GetBlockB, BlockOfHigherOrder=GenerateBlockC, Crop=24)
         GenerateBlockA = functools.partial(GetBlockA, BlockOfHigherOrder=GenerateBlockB, Crop=28)
         HighResolutionPatterns = GenerateBlockA(HiFreqGuess)
         Model = models.Model(inputs=HiFreqGuess, outputs=HighResolutionPatterns)
    Model = multi_gpu_model(Model, gpus=gpus)
    Model.compile(loss='logcosh', optimizer=optimizers.Adam(lr=lr, decay=0))
    return Model