from torch.utils.data import Dataset
import glob,os
import pickle
import numpy as np
import matplotlib.pyplot as plt

PRINT_FLAG = False
class ECG_Multilead_Dataset(Dataset):
    def __init__(self, root_dir= None, transform = None, partial_upload=False):
        super().__init__()
        self.data=[]
        self.data_info=[]
        self.transform= transform


        if root_dir== None:
            self.dataset_path=os.getcwd()+'\\Chineese_database\\'
        else:
            self.dataset_path=root_dir

        for file in glob.glob(self.dataset_path+"*.pkl"):
            unpickled_data=self.unpickle_ECG_data(file=file)
            list_shape=np.shape(unpickled_data)
            if len(list_shape)>1:
                self.data=self.data+unpickled_data
            else:
                self.data_info.append(unpickled_data)
            
            if (partial_upload):
                break
        self.samples = self.data
        data=[d[0] for d in self.data]
        self.data=data
        self.target=[d[1] for d in self.data]
        if (PRINT_FLAG):
            print(f'Uploaded data, size of {np.shape(self.data)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.samples[idx])
        return self.samples[idx]

    def plot(self,idx):
        Leads=['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        Long_lead_type='Lead2'
        item_to_plot=self.samples[idx]
        fig, axes = plt.subplots(nrows=6, ncols=2)
        fig.suptitle(f'Record number {idx}, Is AFIB: {item_to_plot[1]}')
        titles=['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        B=item_to_plot[0][0]
        for ax, cntr in zip(axes.flatten(),range(12)):
            ax.plot(B[cntr,:],linewidth=1.0)
            ax.set(title=titles[cntr])
        plt.plot()
        plt.show()
        return


    def unpickle_ECG_data(self,file='ECG_data.pkl'):
        with open(file, 'rb') as fo:
            pickled_data = pickle.load(fo, encoding='bytes')
        if (PRINT_FLAG):
            print(f'Loaded data with type of: {type(pickled_data)}')
        return pickled_data  

