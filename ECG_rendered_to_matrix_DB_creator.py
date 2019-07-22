import numpy as np
import cv2
from ECG_multi_lead_dataloader import *
from ECG_rendered_multilead_dataloader import *
import matplotlib.pyplot as plt
from ECG_pickling import *
import math


def pair_corresponding_items_from_both_datasets(Matrix_db_path,Rendered_db_path,record_id_start,record_id_end):
    Matrix_obj=ECG_Multilead_Dataset(root_dir=Matrix_db_path,transform= None)
    # Matrix_obj.plot(1)
    Rendered_obj=ECG_Rendered_Multilead_Dataset(root_dir=Rendered_db_path,transform= None,partial_upload=False)
    # Rendered_obj.plot(0)
    assert len(Matrix_obj)==len(Rendered_obj)
    Paired_records=[]
    for cntr in range(record_id_start,record_id_end-1):
        Paired_records.append((Rendered_obj[cntr][0],Matrix_obj[cntr+1][0]))
    return Paired_records



if __name__ == "__main__":
    ## Definitions
    Path_matrix_db=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data'+'\\'
    Path_rendered_db=Path_matrix_db+'Rendering_to_class_db\\'            
    Database_storage_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data\Rendering_to_matrix_db'+'\\'
    print(f'Matrix database path is : {Path_matrix_db}')
    print(f'Rendered database path is : {Path_rendered_db}')
    Records_to_process=41828
    pickle_every=1000
    temp_range=[41]
    for record_cntr in temp_range:#temp_range:#range(math.ceil(Records_to_process/pickle_every)):
        print(f'Processing batch # {record_cntr}')
        Out=pair_corresponding_items_from_both_datasets(Path_matrix_db,Path_rendered_db,record_cntr*pickle_every,
            min(Records_to_process,(record_cntr+1)*pickle_every))
        pickling_filename=Database_storage_path+'Rendered_to_matrix_db'+str(record_cntr)+'.pkl'
        pickle_ECG_data(Out,file=pickling_filename)
        print(f'Pickled: file {pickling_filename}')
    print('Finished')


