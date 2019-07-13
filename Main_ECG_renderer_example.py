import numpy as np
import cv2
from ECG_renderer import *
from ECG_multi_lead_dataloader import *

ECG_test=ECG_Multilead_Dataset(root_dir=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data'+'\\',transform= None, partial_upload=True) 
for cntr in range(15):
    draw_ECG_multilead_vanilla(ECG_test[cntr][0])
print('Finished')