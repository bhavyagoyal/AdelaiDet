import cv2
BASE = 'dircocorepeat'
model_count = 4

lines = open('files.txt','r').readlines()
#lines = open('filescoco.txt','r').readlines()
lines = [x.strip() for x in lines]

for line in lines:
    print(line)
    images = []
    for i in range(model_count):
        im = cv2.imread(BASE+str(i)+'/'+line)
        siz = im.shape
        images.append(im[:,:im.shape[1]//2,:])
    im = cv2.imread(BASE+'/'+line)
    images.append(im[:,:im.shape[1]//2,:])
    im = cv2.imread(BASE+'gt/'+line)
    images.append(im[:,im.shape[1]//2:,:])
    #images = cv2.vconcat([images,im])
    images = cv2.hconcat(images)
    cv2.imwrite(BASE+'comb/'+line.replace('png', 'jpg'), images)
    



