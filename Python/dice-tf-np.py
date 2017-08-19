import numpy as np
import keras.backend as K

#
# 2017-08-19
#   Перестроил реализацию на метод ZFTurbo : +1.0 к знаменателю и числителю. Более корректно работает на пустых картинках.
#                                            Поэтому на крайних/вырожденных значениях возможна разумная неточность
#   На пустых картинках возможно зависание (==0, nan для x=0,  g = 0, s <> 0 (10-200))
#
#   
#
def dicePP (truth, pred, th=None) :
    
    g, s = K.sum(truth,axis=(1,2)), K.sum(pred,axis=(1,2))
    x    = K.sum(truth*pred,axis=(1,2))
    return(K.mean(K.clip((2.0*x+1.0)/(g+s+1.0), K.epsilon(), 1.0)))
    #return((1.0 if K.sum(g+s)==0.0 else 2.0*x/(g+s+K.epsilon())))
    
import tensorflow as tf
def dice01 (truth, pred_, th=0.5) :
    
    #thPred = K.greater(pred_,th)
    
    pred   = tf.to_float(K.greater(pred_,th))
    
    return (dicePP(truth,pred))
    
def dicePP01 (truth, pred, th=0.5) : return((dice01(truth,pred,th=th)+dicePP(truth,pred,th=th))/2.0)

def dice (truth, pred, th=0.5) : 
    return(dice01(truth,pred,th=th))
#    return((dice01(truth,pred,th=th)+dicePP(truth,pred,th=th))/2.0)

def diceNP01 (truth,pred_, th=0.5, printOK=False) :
    pred = pred_.copy()
    pred[pred>th]  = 1.0
    pred[pred<=th] = 0.0
    return(diceNPPP(truth,pred,th=th,printOK=printOK))
    #return((1.0 if K.sum(g+s)==0.0 else 2.0*x/(g+s+K.epsilon())))

def diceNPPP (truth,pred, th=0.5, printOK=False) :
    g, s = truth.sum(axis=(1,2)), pred.sum(axis=(1,2))
    x    = (truth*pred).sum(axis=(1,2))
    if printOK : print('x={} g={} s={}'.format(x,g,s))
    return(((2.0*x+1.0)/(g+s+1.0)).mean())
    #return((1.0 if K.sum(g+s)==0.0 else 2.0*x/(g+s+K.epsilon())))

def diceNP (truth,pred, printOK=False) :
    return(diceNP01(truth,pred))
#    return(diceNPPP(truth,pred))
#    return((diceNPPP(truth,pred)+diceNP01(truth,pred))/2.0)

def lossdice (truth,pred) :
    return(1.0 - dice(truth,pred))

import keras.losses as losses

def lossXdice (truth,pred) :
    return(losses.binary_crossentropy(truth,pred) - K.log(dicePP01(truth,pred)))

def lossYdice (truth,pred) :
    return( - K.log(dicePP01(truth,pred) ))