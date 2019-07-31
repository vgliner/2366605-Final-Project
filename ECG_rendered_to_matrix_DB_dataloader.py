from torch.utils.data import Dataset
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class ECG_Rendered_to_matrix_Dataset(Dataset):
    # Convention   [n , height, width, color channel] 
    def __init__(self, root_dir=None, transform=None, partial_upload=False):
        super().__init__()
        self.data = []
        self.data_info = []
        self.transform = transform
        self.last_chunk_uploaded_to_memory = 1
        self.partial_upload = partial_upload

        if root_dir is None:
            self.dataset_path = os.getcwd()+'\\Chineese_database\\'
        else:
            self.dataset_path = root_dir

        for indx, file in enumerate(glob.glob(self.dataset_path+"*.pkl")):
            unpickled_data = self.unpickle_ECG_data(file=file)
            list_shape = np.shape(unpickled_data)
            self.samples = unpickled_data
            if indx == 0:  # (partial_upload) and
                break
        print(f'Uploaded data, size of {np.shape(self.samples)}')

    def __len__(self):
        if self.partial_upload:
            return len(self.samples)
        else:
            return 41829

    def __getitem__(self, idx):
        chunk_number = idx//1000+1
        if chunk_number == self.last_chunk_uploaded_to_memory:
            if self.transform:
                sample = self.transform(self.samples[idx % 1000])
            else:
                sample = self.samples[idx % 1000]
            return sample
        else:
            file = self.dataset_path+'Rendered_data'+str(max(1, idx//1000+1))+'.pkl'
            unpickled_data = self.unpickle_ECG_data(file=file)
            self.last_chunk_uploaded_to_memory = max(1, idx//1000+1)
            self.samples = unpickled_data
            if self.transform:
                sample = self.transform(self.samples[idx % 1000])
            else:
                sample = self.samples[idx % 1000]
            return sample

    @staticmethod
    def unpickle_ECG_data(file='ECG_data.pkl'):
        with open(file, 'rb') as fo:
            pickled_data = pickle.load(fo, encoding='bytes')
        print(f'Loaded data with type of: {type(pickled_data)}')
        return pickled_data  

    def plot(self, idx):
        item_to_show = self.__getitem__(idx)
        ax1 = plt.subplot(211)
        plt.imshow(item_to_show[0])
        ax2 = plt.subplot(223)
        plt.imshow(item_to_show[1][0])
        ax3 = plt.subplot(224)
        plt.plot(np.squeeze(item_to_show[1][1]))
        ax1.title.set_text('Image')
        ax2.title.set_text('Matrix')
        ax3.title.set_text('Long lead')
        plt.show()
        return


if __name__ == "__main__":
    print('Main is running')
    Rendered_to_matrix_db_path = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data\Rendering_to_matrix_db'+'\\'
    ECG_test = ECG_Rendered_to_matrix_Dataset(root_dir=Rendered_to_matrix_db_path, transform=None)
    for sample_cntr in range(4):
        ECG_test.plot(sample_cntr)
