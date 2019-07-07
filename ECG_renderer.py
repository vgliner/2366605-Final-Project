import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

def draw_ECG_multilead_vanilla(ECG_info: tuple):
    Leads = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL',
             'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    fig, axes = plt.subplots(nrows=3, ncols=4)
    titles = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL','aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    B=ECG_info[0]
    cntr_trans=[0,3,6,9,1,4,7,10,2,5,8,11]
    for ax, cntr in zip(axes.flatten(), range(12)):
        ax.plot(B[cntr_trans[cntr], :], linewidth=2.0,color='black')
        ax.set(title=titles[cntr_trans[cntr]])
        ax.axis('off')

    plt.plot()
    plt.show()
    print('Here')
