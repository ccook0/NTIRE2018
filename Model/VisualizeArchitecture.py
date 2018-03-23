import keras
import Model

Model = Model.GetModel(lr=0, gpus=2)
Model.summary()
keras.utils.plot_model(Model,to_file='Architecture.png',show_shapes=True,show_layer_names=False)