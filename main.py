import numpy as np
from numpy import linalg as LA
from PIL import Image
import os
import matplotlib.pyplot as plt

if not os.path.exists("./img/"):
    os.makedirs("./img/")
    print("Auto create ./img/ file，put the image inside it")

#Just for showing SVD calc in python
def SVD(A):
    A=np.mat(A)
    m,n=np.shape(A)
    if n>m:
        A=A.T
    ATA=A.T.dot(A)
    sigma_nn,V=LA.eig(ATA)
    sigmaNegative=np.array((sigma_nn<0.0).nonzero())[0]
    sigma_nn[sigmaNegative]*=-1
    V[sigmaNegative]*=-1
    sortedSigma__nnIndex=np.argsort(sigma_nn)[::-1]
    sigma_nn=sigma_nn[sortedSigma__nnIndex]
    V=V[:,sortedSigma__nnIndex]
    Sigma=np.mat(np.diag(sigma_nn))*10
    U=A.dot(V).dot(Sigma.I)

    return U,np.array(np.diag(Sigma)),V.T

def svWeight(Sigma,threshold):
    for k in range(len(Sigma)):
        if np.sum(Sigma[:k])/np.sum(Sigma) >= threshold:
            return k

def imgCompress(img,threshold):
    U,Sigma,VT=LA.svd(img)
    k=svWeight(Sigma,threshold)
    reChannel=U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
    Sigma=np.diag(Sigma)

    reChannel[reChannel<0]=0
    reChannel[reChannel>255]=255
    return np.rint(reChannel).astype('uint8')

def imgRebuild(filepath,filename,threshold):
    img=Image.open(filepath+filename,'r')
    A=np.array(img)
    R0=A[:,:,0]
    G0=A[:,:,1]
    B0=A[:,:,2]
    m,n=np.shape(R0)
    if n>m:
        R=imgCompress(R0.T,threshold)
        G=imgCompress(G0.T,threshold)
        B=imgCompress(B0.T,threshold)
    else:
        R=imgCompress(R0, threshold)
        G=imgCompress(G0, threshold)
        B=imgCompress(B0, threshold)
    newImg=np.stack((R,G,B),axis=2)
    name=filename.split('.')[0]
    newFile=f"{name}_{threshold*100}%.jpg"
    Image.fromarray(newImg).save(filepath+newFile)
    img=Image.open(filepath+newFile)
    img.show()

def show_compression_grid(filepath, filename, threshold_list):

    img = Image.open(filepath + filename).convert('RGB')
    A = np.array(img)

    channels_svd = []
    for i in range(3):
        U, sigma, VT = LA.svd(A[:, :, i])
        channels_svd.append((U, sigma, VT))

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    plt.suptitle(f"SVD Compression Comparison: {filename}\n\n", fontsize=20)

    for i, threshold in enumerate(threshold_list):
        reconstructed_channels = []

        for U, sigma, VT in channels_svd:

            k = svWeight(sigma, threshold)
            re_channel = U[:, :k] @ np.diag(sigma[:k]) @ VT[:k, :]


            re_channel = np.clip(re_channel, 0, 255).astype('uint8')
            reconstructed_channels.append(re_channel)


        full_img = np.stack(reconstructed_channels, axis=2)

        axes[i].imshow(full_img)
        axes[i].set_title(f"Threshold: {threshold * 100}% (k={k})")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

try:
    filename='KoalaBear.jpg' #put your image filename
    thresholdList=[0.01,0.05,0.1,0.3,0.5,0.6,0.7,0.8,0.9] #how many % you want to compress
    show_compression_grid("./img/", filename, thresholdList)
except KeyboardInterrupt:
    print("Exit Succesfully!")

