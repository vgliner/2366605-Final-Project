import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import scipy.ndimage


def draw_ECG_multilead_vanilla(ECG_info: tuple):
    Leads = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL',
             'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    img = plt.imread("640px-ECG_paper_12_leads.svg.png")
    #img=scipy.ndimage.zoom(img, 10, order=0)
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    #plt.show()
    image_pixels_ratio=[6000,2500] # (X,Y)
    fig, ax = plt.subplots()
    y_scaling_factor=image_pixels_ratio[1]//6
    ax.imshow(img, extent=[0, image_pixels_ratio[0], 0, image_pixels_ratio[1]])
    ax.plot(ECG_info[1]*y_scaling_factor,  linewidth=1, color='black')
    B=ECG_info[0]
    cntr_trans=[0,3,6,9,1,4,7,10,2,5,8,11]
    for cntr in range(4):
        ax.plot(np.linspace(cntr*image_pixels_ratio[0]//5,cntr*image_pixels_ratio[0]//5+len(B[cntr_trans[cntr], :]),num=len(B[cntr_trans[cntr], :])),B[cntr_trans[cntr], :]*y_scaling_factor+4/5*image_pixels_ratio[1],  linewidth=1, color='black')

    ax.axis('off')

    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=4)
    titles = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL','aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    B=ECG_info[0]
    for ax, cntr in zip(axes.flatten(), range(12)):
        ax.plot(B[cntr_trans[cntr], :], linewidth=2.0,color='black')
        ax.set(title=titles[cntr_trans[cntr]])
        ax.axis('off')

    plt.plot()
    plt.show()
    print('Here')
