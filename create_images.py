
import cv2
BASE = 'dir'
model_count = 6

lines = open('files.txt','r').readlines()
lines = [x.strip() for x in lines]

for line in lines:
    images = []
    for i in range(model_count):
        im = cv2.imread(BASE+str(i)+'/'+line)
        siz = im.shape
        images.append(im[:,:im.shape[1]//2,:])
    im = cv2.imread(BASE+'/'+line)
    siz = im.shape
    #images = cv2.vconcat([images,im])
    images.append(im[:,:im.shape[1]//2,:])
    images.append(im[:,im.shape[1]//2:,:])
    images = cv2.hconcat(images)
    cv2.imwrite(BASE+'comb/'+line, images)
#for i in range(model_count):
    



