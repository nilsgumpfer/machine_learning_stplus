import numpy as np

''' Example Usage:

    model.fit(x=X_train,
                  y=Y_train,
                  batch_size=params['batch_size'],
                  epochs=params['number_epochs'],
                  shuffle=params['shuffle'],
                  validation_data=(X_val, Y_val),
                  callbacks=[custom_callback],
                  class_weight=derive_class_weight(Y_train))
              
'''
def derive_class_weight(y_train):
    cw = {i: 1 - (np.sum(y_train[..., i]) / len(y_train)) for i in range(len(y_train[0]))}
    return cw