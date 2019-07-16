#TODO: Adjust the file to fit the actual database


#%% ECG Chineese Database tester  
import sklearn as sk
import matplotlib.pyplot as plt
import torchvision.transforms as tvtf
from ECG_rendered_multilead_dataloader import *
import transforms as tf
import torchvision
import torch
import knn_classifier as knn
import os


# Define the transforms that should be applied to each ECG record before returning it
tf_ds = tvtf.Compose([
    tf.ECG_tuple_transform(-1) # Reshape to 1D Tensor
])

ECG_test=ECG_Rendered_Multilead_Dataset(root_dir=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data\Rendering_to_class_db'+'\\',
    transform= None,partial_upload=True) # For KNN demo
########   Example how to access the data (Uncomment if necessary) ##############
sample_test=ECG_test[2] #Taking for example record number 2 (Starting from zero)
print(f'Size of an image of the data of 12 leads :{np.shape(sample_test[0])}')
print(f'Is the example record AFIB: {sample_test[1]}')
plt.imshow(sample_test[0])
plt.show()





# Define how much data to load (only use a subset for speed)
num_train = 900
num_test = 99
batch_size = 1000

# Training dataset & loader
ds_train = tf.SubsetDataset(ECG_test, num_train)  #(train=True, transform=tf_ds)
dl_train = torch.utils.data.DataLoader(ds_train,batch_size= batch_size,shuffle=False)

# Test dataset & loader
ds_test = tf.SubsetDataset(ECG_test, num_test, offset= num_train)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size)

# for batch_indx, sample in enumerate(dl_train):
#     print(sample)


# Get all test data to predict in one go
test_iter = iter(dl_test)
x_test, y_test = test_iter.next()



# Test kNN Classifier
knn_classifier = knn.KNNClassifier(k=10)
knn_classifier.train(dl_train)
y_pred = knn_classifier.predict(x_test)

# Calculate accuracy
accuracy = knn.accuracy(y_test, y_pred)
print(f'Accuracy: {accuracy*100:.2f}%')