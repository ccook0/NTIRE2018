import keras




Model = keras.models.load_model('Model_8th.h5')
#trained 300 + 300 iterations
Model.save_weights('w_8th.h5')