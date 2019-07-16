import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import numpy as np
import scipy.ndimage
import time
import os


def draw_ECG_multilead_vanilla(ECG_info: tuple):
    Leads = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL',
             'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    img = plt.imread("640px-ECG_paper_12_leads.svg.png")
    pixels_to_mV= 150//10 # 147 pixels /10mV
    pixels_to_sec= (384.76-93.6)//(0.2*3) # 
    image_pixels_ratio=[6000,2500] # (X,Y)
    fig, ax = plt.subplots()
    # fig._tight=True
    ax._frameon=False
    y_scaling_factor=image_pixels_ratio[1]//7
    ax.imshow(img, extent=[0, image_pixels_ratio[0], 0, image_pixels_ratio[1]])
    x_cal_signal=np.linspace(0,0.2*pixels_to_sec,num=np.floor(0.2*pixels_to_sec))
    y_cal_signal=np.ones_like(x_cal_signal)*pixels_to_mV*10+ECG_info[1][0][1]*y_scaling_factor+pixels_to_mV*10
    y_cal_signal[-2:]=ECG_info[1][0][1]*y_scaling_factor+pixels_to_mV*10
    ax.plot(x_cal_signal,y_cal_signal,linewidth=0.5, color='black')        

    x_axis_long_lead=np.arange(len(ECG_info[1][0]))
    x_axis_long_lead+=len(x_cal_signal)
    ax.plot(x_axis_long_lead,ECG_info[1][0]*y_scaling_factor+pixels_to_mV*10,  linewidth=0.5, color='black')
    B=ECG_info[0]
    cntr_trans=[0,3,6,9,1,4,7,10,2,5,8,11]
    for extrnl_cntr in range(3):
        # Plot a calibration signal
        x_cal_signal=np.linspace(0,0.2*pixels_to_sec,num=np.floor(0.2*pixels_to_sec))
        y_cal_signal=np.ones_like(x_cal_signal)*pixels_to_mV*10+B[cntr_trans[extrnl_cntr*4]][0]*y_scaling_factor+(4-extrnl_cntr)/4.5*image_pixels_ratio[1]-pixels_to_mV*20
        y_cal_signal[-2:]=0+B[cntr_trans[extrnl_cntr*4]][0]*y_scaling_factor+(4-extrnl_cntr)/4.5*image_pixels_ratio[1]-pixels_to_mV*20
        ax.plot(x_cal_signal,y_cal_signal,linewidth=0.5, color='black')        
        for cntr in range(4):
            ax.plot(np.linspace(cntr*image_pixels_ratio[0]//4.7+len(y_cal_signal),cntr*image_pixels_ratio[0]//4.7+len(B[cntr_trans[cntr], :])+len(y_cal_signal),
            num=len(B[cntr_trans[cntr], :])),B[cntr_trans[extrnl_cntr*4+cntr], :]*y_scaling_factor+(4-extrnl_cntr)/4.5*image_pixels_ratio[1]-pixels_to_mV*20,
            linewidth=0.5, color='black')

    ax.axis('off')
    plt.tight_layout()
    # plt.get_current_fig_manager().window.state('zoomed')
    # plt.gcf().delaxes(plt.gca())
    # plt.gcf().add_axes(ax)
    try:
        plt.savefig('test.png',dpi=300,quality=95,transparent= True,pad_inches=0.0,bbox_inches='tight',facecolor=None,edgecolor=None)
    except:
        time.sleep(1)
        os.remove('test.png')
        plt.savefig('test.png',dpi=300,quality=95,transparent= True,pad_inches=0.0,bbox_inches='tight',facecolor=None,edgecolor=None)
    #plt.show()

    # canvas = FigureCanvas(fig)
    # ax = fig.gca()
    # canvas.draw()       # draw the canvas, cache the renderer
    # s, (width, height) = canvas.print_to_buffer()
    # X = np.fromstring(s, np.uint8).reshape((height, width, 4))  # The format is RGBA (A- alpha)
    plt.close("all")

    X=mpimg.imread('test.png')
    # plt.imshow(X)
    # plt.plot()
    return X[:,:,0:3]


def find_image_frames(Input_image):
    image_shape=np.shape(Input_image)
    pivot_X=image_shape[1]//2
    pivot_Y=image_shape[0]//2
    min_x=0
    max_x=image_shape[1]
    min_y=0
    max_y=image_shape[0]
    B=(Input_image[:,:,0]<1.) *(Input_image[:,:,1]<1.) *(Input_image[:,:,2]<1.) 
    while (B[pivot_Y,min_x]==False): # Find min X
        min_x+=1
    while (B[pivot_Y,max_x-1]==False): # Find min X
        max_x-=1    
    while (B[min_y,pivot_X]==False): # Find min X
        min_y+=1
    while (B[max_y-1,pivot_X]==False): # Find min X
        max_y-=1       
    
    
    print(f'{min_x} {max_x} {min_y} {max_y}')

    return (min_x, max_x, min_y, max_y)