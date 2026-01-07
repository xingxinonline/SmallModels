from turtle import shape
from PIL import Image
import numpy as np
import time
import cv2

def my_open_jpg(filename):
    return Image.open(filename)

def my_jpg2yuv(im_jpg):
    r,g,b=im_jpg.split()

    width,height=im_jpg.size
    im_new=np.arange(int(width*height*3/2))
    r = list(r.getdata())
    g = list(g.getdata())
    b = list(b.getdata())
    for i in range(height):
        for j in range(width):
            index = i*width+j
            R = r[index]
            G = g[index]
            B = b[index]

            Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16
            U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128
            V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128
            if((i&0x01 == 0)  and (j&0x01 ==0)):
                im_new[int(width*height+i/2*width/2+j/2)] = V
                im_new[int(width*height*5/4+i/2*width/2+j/2)] = U
            im_new[index]=Y

    return im_new
def my_save_yuv(filename,im_new):
    #use the numpy to write the data to file
    fp = open(filename,"wb")
    data=np.array(im_new,"B")
    data.tofile(fp)
    fp.close()
    print("save yuv file %s successfully" % filename)





def GetImage():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    count = 0

    while count < 3000:
        ret, img_raw = cap.read()
        img_raw = cv2.resize(img_raw, (640, 480))
        cv2.imwrite("image/" + str(count) + ".jpg",img_raw)
        print(time.time())
        count +=1
    cap.release()




def ResizeChangeYuv(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image,(1280,960))
    cv2.imwrite(filename,image)

    im = my_open_jpg(filename)
    im_yuv = my_jpg2yuv(im)
    my_save_yuv(filename[:-4] + ".yuv",im_yuv)


def ChangeYuv():
    for count in range(183,281):
        filename = "image/"+str(count)+".jpg"
        im = my_open_jpg(filename)
        im_yuv = my_jpg2yuv(im)
        my_save_yuv(filename[:-4] + ".yuv",im_yuv)

def Resize(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image,(320,240))
    cv2.imwrite("123.jpg",image)


def pic2RGB():
    image = Image.open('face.jpg')


    r,g,b=image.split()

    im_new=np.arange(int(112*112*3))
    r = list(r.getdata())
    g = list(g.getdata())
    b = list(b.getdata())
    height = 112
    width = 112
    for i in range(height):
        for j in range(width):
            im_new[i*width+j] = r[i*width+j]
            im_new[width*height + i*width+j] = g[i*width+j]
            im_new[2*width*height + i*width+j] = b[i*width+j]
    
    fp = open("face.rgb","wb")
    data=np.array(im_new,"B")
    data.tofile(fp)
    fp.close()



if __name__ == "__main__":
    #GetImage()
    #ResizeChangeYuv("20020724img_373.jpg")
    #ChangeYuv()
    #Resize("20020724img_373.jpg")
    pic2RGB()