# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn
import numpy as np
import  cv2
import matplotlib.pyplot as plt

def get_bilinear_filter(upscale_factor):
    #根据upscale_factor计算kernel_size
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5
    weight = np.zeros([kernel_size, kernel_size])
    for x in range(kernel_size):
        for y in range(kernel_size):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                        1 - abs((y - centre_location) / upscale_factor))
            weight[x, y] = value
    return weight

def get_bilinear(upscale_factor):
    """
    :param upscale_factor:  上采样比例
    :return:   nn.ConvTranspose2d object
    """

    #根据upscale_factor计算kernel_size
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    # 根据kernel_size按照逆过程（下采样）计算padding
    padding=(kernel_size-upscale_factor)//2   # (upscale_factor*size-kernel_size+2*padding)/upscale_factor+1=size
    weight=torch.tensor(get_bilinear_filter(upscale_factor)).view([1,1,kernel_size,kernel_size])
    conv=nn.ConvTranspose2d(1,1,kernel_size=kernel_size,stride=upscale_factor,padding=padding,bias=False)
    conv.weight=nn.Parameter(weight)
    return conv

def plt_show_imgs(imgs,title=None):
    assert isinstance(imgs,(list,tuple))
    #fig=plt.figure()
    length=len(imgs)
    for i in range(length):
        plt.subplot(1,length,i+1)
        plt.imshow(imgs[i])
    plt.show()
if __name__=="__main__":

    image=cv2.imread("./1.jpg",flags=0) #读灰度图
    ipt_arr=np.array(image,np.float)
    ipt_tensor=torch.tensor(ipt_arr[None][None])
    upscale2=get_bilinear(2).eval()
    upscale3=get_bilinear(3).eval()
    upscale4=get_bilinear(4).eval()
    opt_tensor2=upscale2(ipt_tensor)
    opt_tensor3=upscale3(ipt_tensor)
    opt_tensor4=upscale4(ipt_tensor)
    opt_arr2=opt_tensor2.detach().numpy().squeeze()
    opt_arr3=opt_tensor3.detach().numpy().squeeze()
    opt_arr4=opt_tensor4.detach().numpy().squeeze()
    print(ipt_arr.shape,opt_arr2.shape,opt_arr3.shape,opt_arr4.shape,)
    plt_show_imgs([image,opt_arr2,opt_arr3,opt_arr4])


