import numpy as np
import cv2
from ECG_renderer import *
from ECG_multi_lead_dataloader import *

ECG_test=ECG_Multilead_Dataset(root_dir=os.getcwd()+'\\Chineese_database\\',transform= None, partial_upload=True) 

draw_ECG_multilead_vanilla(ECG_test[2][0])
print('Finished')