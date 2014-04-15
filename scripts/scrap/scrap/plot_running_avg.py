# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 22:50:18 2014

@author: tegjyot
"""
import matplotlib as mp
def plot_running_accuracy(y_true,y_pred):
        avg_accuracy=[]
        running_accuracy=0
        for i in range(1,len(y_true)):
            if y_true[i-1]==y_pred[i-1]:
                current_accuracy=1
            else:
                current_accuracy=0
            running_accuracy=running_accuracy+current_accuracy
            avg_accuracy.append(float(running_accuracy)/i)
        x=range(1,len(avg_accuracy)+1)
        #x=range(1,len(chunk_accuracy_list)+1,1000)
        mp.pyplot.plot(x,avg_accuracy)
        #mp.pyplot.show()
        