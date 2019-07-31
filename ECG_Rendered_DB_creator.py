import numpy as np
import cv2
from ECG_renderer import *
from ECG_multi_lead_dataloader import *
import matplotlib.pyplot as plt
from ECG_pickling import *

ECG_test = ECG_Multilead_Dataset(root_dir=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data' + '\\',
                                 transform=None, partial_upload=False)
ECG_Classified_Rendered = []
pickle_every = 1000
Pickle_path = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data\Rendering_to_class_db' + '\\'
Current_image = draw_ECG_multilead_vanilla(ECG_test[1459][0])  # 369
K = find_image_frames(Current_image)
plt.imshow(Current_image[K[2]:K[3], K[0]:K[1] - 200, :])
plt.show()

for cntr in range(0, 1001):  # len(ECG_test)
    Current_image = draw_ECG_multilead_vanilla(ECG_test[cntr][0])
    ECG_Classified_Rendered.append((Current_image[K[2]:K[3], K[0]:K[1] - 200, :], ECG_test[cntr][1]))
    print(f'Processed image number {cntr} with shape of {np.shape(Current_image)}')
    try:
        if (cntr > 0) and (cntr % pickle_every == 0):
            print(f'Writing fo file..., counter is {cntr}')
            pickle_ECG_data(ECG_Classified_Rendered[-pickle_every:],
                            Pickle_path + 'Rendered_data' + str(cntr // pickle_every) + '.pkl')
            ECG_Classified_Rendered = []
            last_cntr = cntr

        if (cntr + 1) // len(ECG_test):
            print(f'Writing fo file..., counter is {cntr}')
            pickle_ECG_data(ECG_Classified_Rendered[-pickle_every:],
                            Pickle_path + 'Rendered_data' + str(cntr // pickle_every + 1) + '.pkl')
            ECG_Classified_Rendered = []
            last_cntr = cntr
    except:
        if (cntr > 0) and (cntr % pickle_every == 0):
            print(f'Writing fo file..., counter is {cntr}')
            pickle_ECG_data(ECG_Classified_Rendered[-pickle_every:],
                            Pickle_path + 'Rendered_data' + str(cntr // pickle_every) + '.pkl')
            ECG_Classified_Rendered = []
            last_cntr = cntr
if cntr % pickle_every:
    pickle_ECG_data(ECG_Classified_Rendered[
                    -len(ECG_Classified_Rendered) + pickle_every * (len(ECG_Classified_Rendered) // pickle_every):len(
                        ECG_Classified_Rendered)],
                    Pickle_path + 'Rendered_data' + str(len(ECG_Classified_Rendered) // pickle_every) + '.pkl')

print('Finished')
