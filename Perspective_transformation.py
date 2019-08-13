import random
from PIL import Image
import torchvision.transforms.functional as F
import torchvision
from ECG_rendered_to_matrix_DB_dataloader import *
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from ECG_multi_lead_dataloader import *




class RandomPerspective(object):
    """Performs Perspective transformation of the given PIL Image randomly with a given probability.
    Args:
        interpolation : Default- Image.BICUBIC
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.
        Returns:
            PIL Image: Random perspectivley transformed image.
        """
        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return F.perspective(img, startpoints, endpoints, self.interpolation)
        return img

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.
        Args:
            width : width of the image.
            height : height of the image.
        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

PHASE =1

if __name__=="__main__":
    target_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'
    ECG_test=ECG_Multilead_Dataset(target_path)
    B=ECG_test[2]
    B.plot(2)

    print('Evaluating')
    ECG_test = ECG_Rendered_to_matrix_Dataset(root_dir=target_path, transform=None, partial_upload=False)  
    K=ECG_test[2]
    print(f'ECG shape is : {np.shape(K[0])} ')
    load_file='Backgrounds\\'
    ECG_shape=np.shape(K[0])
    if PHASE==1:
        print('Executing phase 1')
        for image_cntr in range(1,11):
            load_file='Backgrounds\\'
            load_file=load_file+str(image_cntr)+'.jpg'
            face = misc.imread(load_file)
            background_shape=np.shape(face)
            ratio=[aItem/bItem for aItem, bItem in zip(ECG_shape, background_shape)]
            print(f'Shape of {image_cntr} is: {np.shape(face)}')
            # plt.imshow(face)
            # plt.show()
            # for cntr in range(4):
            #     strt=time.time()
            #     face1=ndimage.zoom(face,2,order=cntr)
            #     face1=face1[:,:,0:3]
            #     stp=time.time()
            #     print(f'Elapsed time : {stp-strt}, counter : {cntr} ')
            #     plt.imshow(face1)
            #     plt.show()                                                


            # backgroundImage=Image.open(load_file)            

        print('Finished phase 1')

    ## Uploading image
    Im=Image.open("test.png")
    Im.show()
    for cntr in range(10):
        Current_ECG=ECG_test[cntr]
        ECG_image=Current_ECG[0]
        # plt.imshow(ECG_image)
        # plt.show()
        # Convert PIL to numpy  -> pix = numpy.array(pic)
        # Convert numpy to PIL -> im = Image.fromarray(np.uint8(ECG_image*255))

#TODO: Save backgrounds as numpy array

        ECG_image_PLL=Image.fromarray(ECG_image)
        ECG_image_PLL.show()
        load_file=load_file+str(random.randint(1,10))+'.jpg'
        backgroundImage=Image.open(load_file)
        P=RandomPerspective(distortion_scale=0.4, p=0.5, interpolation=3)
        new_img = Image.blend(backgroundImage, Im, 0.5)
        Im2=P(Im)
        Im2.show()

    T=torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)
    print('Finished')

