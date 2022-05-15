from kornia.feature import LoFTR
import cv2
import os
import kornia as K
from kornia_moons.feature import *
import torch
import kornia.feature as KF
import matplotlib.pyplot as plt

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

matcher = LoFTR(pretrained="outdoor")
# correspondences_dict = matcher(input)

BASE_DIR = '/home/karan/kitti/KITTI_tiny/'
def draw_matches(mkpts0, mkpts1,img1, img2, inliers, H):

    ret = draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                'tentative_color': None, 
                'feature_color': (0.2, 0.5, 1), 'vertical': False},
        H=H)
    

    plt.show()

def find_matches(img1_path, img2_path, confidence_threshold = 0.5):
    """
    Extract these features from images:
    1. correspondences : keypoints in image space
        a vector of important positions in image [{x, y}...]
    2. H: fundamental matrix
        Rotational(R), T matrix
    3. inlners: one hot vector of the points that we will use as good matches
        len(inliers) == len(correspondences)
    """
    img1 = load_torch_image(img1_path)
    img2 = load_torch_image(img2_path)
     
    input = {"image0": K.color.rgb_to_grayscale(img1), "image1": K.color.rgb_to_grayscale(img2)}
    correspondences = matcher(input)
    mkpts0 = correspondences['keypoints0'].cpu().numpy()[correspondences['confidence'] > confidence_threshold]
    mkpts1 = correspondences['keypoints1'].cpu().numpy()[correspondences['confidence'] > confidence_threshold]
    print(mkpts0.shape)
    print(mkpts1.shape)
    H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0
    print("hello")
    draw_matches(mkpts0, mkpts1,img1, img2, inliers, H)
    return correspondences, H, inliers

def get_imgs():
    """get data"""
    with open(os.path.join(BASE_DIR, 'kitti_tiny.txt'), "r") as f:
        img_path1, img_path2 = f.readline().rstrip().split(' ')
        
        img1_path = os.path.join(BASE_DIR, img_path1)
        img2_path = os.path.join(BASE_DIR, img_path2)
        
        img1_path = './data/kn_church-2.jpg'
        img2_path = './data/kn_church-8.jpg'
        return img1_path, img2_path
        

img1_path, img2_path = get_imgs()
find_matches(img1_path, img2_path)


