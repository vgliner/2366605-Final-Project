import numpy as np
import cv2
from ECG_renderer import *
from ECG_multi_lead_dataloader import *
import matplotlib.pyplot as plt
from ECG_pickling import *

ECG_test=ECG_Multilead_Dataset(root_dir=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data'+'\\',transform= None, partial_upload=True) 
ECG_Classified_Rendered=[]
pickle_every=1000
Pickle_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data\Rendering_to_class_db'+'\\'

for cntr in range(pickle_every):
    Current_image=draw_ECG_multilead_vanilla(ECG_test[cntr][0])
    ECG_Classified_Rendered.append((Current_image,ECG_test[cntr][1]))
    print(f'Processed image number {cntr} with shape of {np.shape(Current_image)}')
    if (cntr>0) and (cntr % pickle_every==0):
        pickle_ECG_data(ECG_Classified_Rendered[cntr-pickle_every:cntr],Pickle_path+'Rendered_data'+str(cntr % pickle_every)+'.pkl')
        last_cntr = cntr
if (cntr % pickle_every):
    pickle_ECG_data(ECG_Classified_Rendered[-len(ECG_Classified_Rendered)+pickle_every*(len(ECG_Classified_Rendered)// pickle_every):len(ECG_Classified_Rendered)],
    Pickle_path+'Rendered_data'+str(len(ECG_Classified_Rendered) // pickle_every  )+'.pkl')


print('Finished')