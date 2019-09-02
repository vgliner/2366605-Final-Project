import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
For perspective transformation, you need a 3x3 transformation matrix. 
Straight lines will remain straight even after the transformation. 
To find this transformation matrix, you need 4 points on the input image and corresponding points on the output image. 
Among these 4 points, 3 of them should not be collinear. 
Then transformation matrix can be found by the function cv2.getPerspectiveTransform. 
Then apply cv2.warpPerspective with this 3x3 transformation matrix.


Code example
img = cv2.imread('sudokusmall.png')
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

"""

class Perspective_Rendering:
    def __init__(self, rendering_type=None):
        self.rendering_type=rendering_type
        self.backgrounds_path=r'C:\Users\vgliner\OneDrive - JNJ\Documents\GitHub\2366605-Final-Project\Backgrounds'+'\\'

    def Upload_background(self, background_number=-1):
        """ Backgound number should be between 1-10, otherwise is chooses randombly one of the backgrounds
        """
        if background_number<1 or background_number>10:
            background_number=1
            # TODO: Later correct to random number
        img = cv2.imread(self.backgrounds_path+f'{background_number}.jpg')
        rows,cols,ch = img.shape
        # print(f'Loaded background template number {background_number},shape of {rows}, {cols}, {ch}')
        return img



if __name__ == "__main__":
    img = cv2.imread('test.png')
    rows,cols,ch = img.shape
    print(f'Loaded image, {rows}, {cols}, {ch}')
    plt.imshow(img)
    plt.show()
    pts1 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    pts2 = np.float32([[0,0+30],[rows,0+100],[0,cols-30],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(2000,2000))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()
    P=Perspective_Rendering()
    Background_image=P.Upload_background(2)
    plt.imshow(Background_image)
    plt.show()