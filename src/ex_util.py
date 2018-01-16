# -*- coding: utf_8 -*-
''' Utils ''' 

#----------------------------------------------------------
# Imports
#----------------------------------------------------------
from matplotlib     import pyplot


#----------------------------------------------------------
# classes
#----------------------------------------------------------
class Util:
    ''' Utility Class'''
    
    @staticmethod
    def plot_result(hist):
        ''' Show results with plot '''
        
        pyplot.subplot(1, 2, 1)
        pyplot.ylim(0, 1)
        pyplot.plot(hist.history['acc'], label="acc", color="red", linestyle='dotted')
        pyplot.plot(hist.history['val_acc'], label="val_acc", color="red")
        pyplot.legend()
        
        pyplot.subplot(1, 2, 2)
        pyplot.ylim(0, 1)
        pyplot.plot(hist.history['loss'], label="loss", color="red", linestyle='dotted')
        pyplot.plot(hist.history['val_loss'], label="val_loss", color="red")
        pyplot.legend()
        pyplot.show()
