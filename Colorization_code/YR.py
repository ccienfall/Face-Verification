import numpy as np
from PIL import Image
RGB2YUV = (np.array([0.212600,0.715200,0.072200,-0.114572,-0.385428,0.500000,0.500000,-0.454153 ,-0.045847 ])).reshape(3,3)


im = Image.open('202598.jpg')

if im.mode != "RGB":
    im = im.convert("RGB")
new_size = [int(i/1.3) for i in im.size]
im.thumbnail(new_size, Image.ANTIALIAS)
target = np.asarray(im,dtype=float)[3:-3,4:-4,:]
target = target/255.0
target = (np.dot(target.reshape(-1,3),RGB2YUV.transpose())).reshape(160,128,3)
target[:,:,0:1] = 219.0*target[:,:,0:1]+16
target[:,:,1:] = 224.0*target[:,:,1:]+128
target = np.array(target,dtype=np.uint8)


def clip(data,min,max):
    out = data
    out[data<min] = min
    out[data>max] = max
    return out
YUV2RGB = (np.array([1,0,1.57480,1,-0.18733,-0.46813,1,1.85563,0])).reshape(3,3)

target = np.array(target,dtype=float)
target[:,:,0:1] = clip((target[:,:,0:1]-16)/219.0, 0 , 1)
target[:,:,1:] = clip((target[:,:,1:]-128)/224.0, -0.5 , 0.5)
target = np.dot(target.reshape(-1,3),YUV2RGB.transpose())
target = (np.array(255.0*target,dtype=np.uint8)).reshape(160,128,3)

ims = Image.fromarray(target)
ims.show()

