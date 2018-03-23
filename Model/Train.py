import IO
import PatternManipulator
import Model
import CustomCallbacks
from keras import callbacks

GPU = 2
PatternSize = 159
Margin = PatternManipulator.GetMargin([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])
TargetSize = PatternSize - 2 * Margin
batch_size = 3 * GPU
Length = 995124 #2x:934776

Guess = IO.LoadRawBinaryGrayscaleSequenceAdvanced('GuessAugmented4x.bin', PatternSize, PatternSize, Length)
Target = IO.LoadRawBinaryGrayscaleSequenceAdvanced('TargetAugmented4x.bin', TargetSize, TargetSize, Length)

DynamicLearningRateReducer = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=1e-16, patience=10, verbose=True)
AutoSave = CustomCallbacks.ModelCheckpointDetached(filepath='Model4x-{epoch:03d}.h5', verbose=1, period=1)

Block = Model.GetModel(lr=2e-4, gpus=GPU)
Block.summary()
Block.fit(Guess, Target, batch_size=batch_size, epochs=10000, callbacks=[DynamicLearningRateReducer, AutoSave])