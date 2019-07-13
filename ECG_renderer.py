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
    pixels_to_mV= 150//10 # 147 pixels /10mV
    pixels_to_sec= (384.76-93.6)//(0.2*3) # 
    image_pixels_ratio=[6000,2500] # (X,Y)
    fig, ax = plt.subplots()
    y_scaling_factor=image_pixels_ratio[1]//7
    ax.imshow(img, extent=[0, image_pixels_ratio[0], 0, image_pixels_ratio[1]])
    x_cal_signal=np.linspace(0,0.2*pixels_to_sec,num=np.floor(0.2*pixels_to_sec))
    y_cal_signal=np.ones_like(x_cal_signal)*pixels_to_mV*10+ECG_info[1][0][1]*y_scaling_factor+pixels_to_mV*10
    y_cal_signal[-2:]=ECG_info[1][0][1]*y_scaling_factor+pixels_to_mV*10
    ax.plot(x_cal_signal,y_cal_signal,linewidth=1, color='black')        

    x_axis_long_lead=np.arange(len(ECG_info[1][0]))
    x_axis_long_lead+=len(x_cal_signal)
    ax.plot(x_axis_long_lead,ECG_info[1][0]*y_scaling_factor+pixels_to_mV*10,  linewidth=1, color='black')
    B=ECG_info[0]
    cntr_trans=[0,3,6,9,1,4,7,10,2,5,8,11]
    for extrnl_cntr in range(3):
        # Plot a calibration signal
        x_cal_signal=np.linspace(0,0.2*pixels_to_sec,num=np.floor(0.2*pixels_to_sec))
        y_cal_signal=np.ones_like(x_cal_signal)*pixels_to_mV*10+B[cntr_trans[extrnl_cntr*4]][0]*y_scaling_factor+(4-extrnl_cntr)/4.5*image_pixels_ratio[1]-pixels_to_mV*20
        y_cal_signal[-2:]=0+B[cntr_trans[extrnl_cntr*4]][0]*y_scaling_factor+(4-extrnl_cntr)/4.5*image_pixels_ratio[1]-pixels_to_mV*20
        ax.plot(x_cal_signal,y_cal_signal,linewidth=1, color='black')        
        for cntr in range(4):
            ax.plot(np.linspace(cntr*image_pixels_ratio[0]//4.7+len(y_cal_signal),cntr*image_pixels_ratio[0]//4.7+len(B[cntr_trans[cntr], :])+len(y_cal_signal),
            num=len(B[cntr_trans[cntr], :])),B[cntr_trans[extrnl_cntr*4+cntr], :]*y_scaling_factor+(4-extrnl_cntr)/4.5*image_pixels_ratio[1]-pixels_to_mV*20,
            linewidth=1, color='black')

    ax.axis('off')
    # plt.show()

    canvas = FigureCanvas(fig)
    ax = fig.gca()
    canvas.draw()       # draw the canvas, cache the renderer
    s, (width, height) = canvas.print_to_buffer()
    X = np.fromstring(s, np.uint8).reshape((height, width, 4))  
    
    return X

