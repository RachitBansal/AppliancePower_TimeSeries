import keras.backend as K

def r2(y_true, y_pred):
	y_mean = K.sum(y_true, axis=-1)/y_true.shape[0]
	return K.sqrt(1- K.square(y_true-y_pred)/K.square(y_true-y_mean))
