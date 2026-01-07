import numpy as np
import os
import cv2
import torch
import random
import time

import torch.backends.cudnn as cudnn
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

from model.detect_utils import cfg
from model.detect_utils import PriorBox
from model.detect_utils import nms
from model.detect_utils import decode
from model.matlab_cp2tform import get_similarity_transform_for_cv2


import config


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_quant_model(model, pretrained_path, load_to_cpu):
    model.eval()
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    prepared_model = prepare_fx(model, qconfig_dict)
    quantized_model = convert_fx(prepared_model)
    quantized_model = load_model(quantized_model, pretrained_path, load_to_cpu)
    return quantized_model

def CutImage(image,bbox,extend = 0):
    center_x = (bbox[2] + bbox[0])/2
    center_y = (bbox[3] + bbox[1])/2
    half_x = int((bbox[2] - center_x)*(1+extend))
    half_y = int((bbox[3] - center_y)*(1+extend))

    xmax = min(image.shape[1],int(center_x+half_x))
    xmin = max(0,int(center_x-half_x))
    ymax = min(image.shape[0],int(center_y+half_y))
    ymin = max(0,int(center_y-half_y))

    return image[ymin:ymax,xmin:xmax]

def FaceDetect(net,img_raw):
    device = torch.device(config.device)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    scale = torch.Tensor([im_width, im_height, im_width, im_height,
                        im_width, im_height, im_width, im_height,
                        im_width, im_height, im_width, im_height,
                        im_width, im_height])
    scale = scale.to(device)
    loc, conf, iou = net(img)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()

    cls_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]





    iou_scores = iou.squeeze(0).data.cpu().numpy()[:, 0]
    _idx = np.where(iou_scores < 0.)
    iou_scores[_idx] = 0.
    _idx = np.where(iou_scores > 1.)
    iou_scores[_idx] = 1.
    scores = np.sqrt(cls_scores * iou_scores)

    # ignore low scores
    inds = np.where(scores > config.detect_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:config.detect_top_k]
    boxes = boxes[order]
    scores = scores[order]


    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    selected_idx = np.array([0,1,2,3,14])
    keep = nms(dets[:,selected_idx], config.detect_nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:config.detect_keep_topk, :]

    result = []
    for item in dets:
        if(item[2] - item[0] > 5 and item[3] - item[1] > 5):
            result.append(item)

    return np.array(result)

def ImageAlign(src_img,src_pts):
    #ref_pts = [[30.2946, 51.6963],[65.5318, 51.5014],
    #    [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041]]
    ref_pts = [[30.2946+8, 51.6963],[65.5318+8, 51.5014],
        [48.0252+8, 71.7366],[33.5493+8, 92.3655],[62.7299+8, 92.2041]] 
    crop_size = (112, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def LoadVector():
    return np.load(config.recog_vector_path, allow_pickle=True).item()
    
def Face2Vector(model,image):
    img1 = image
    img2 = img1[:, ::-1, :]
    imglist = [img1,img2]
    for i in range(len(imglist)):
        imglist[i] = (imglist[i] - 127.5) / 128.0
        imglist[i] = imglist[i].transpose(2, 0, 1)
    imgs = [torch.from_numpy(i).float().cpu().unsqueeze(0) for i in imglist]
    vector1 = model(imgs[0]).detach().numpy()[0]
    vector2 = model(imgs[1]).detach().numpy()[0]

    vector = np.concatenate([vector1,vector2],axis=0)

    return vector


def SingleFaceRecog(model,image,vector_dict):
    #cv2.imshow("result22",image)
    #cv2.waitKey(100)

    #image = cv2.imread("pic\Li_Ka-shing\Li_Ka-shing_0001.jpg")
    #print(image)
    vector = Face2Vector(model,image)
    #print(vector*128)
    #exit(0)


    
    vector = (vector - vector_dict['mu'])
    vector = vector/np.sqrt(np.sum(np.power(vector, 2)))
    similarity = -1
    name = "unknown"
    for key in vector_dict.keys():
        if key == "mu":
            continue
        
        #print(vector_dict[key],vector)
        similar = np.dot(vector_dict[key],vector)
        
        if ((similar > config.recog_thresh) and (similar > similarity)):
            print(similar,key)
            similarity = similar
            name = key
    
    return name

def FaceRecog(net,image,detect_result):
    output_result = []
    vector_dict = LoadVector()
    
    for result in detect_result:
        source_points = result[4:14]
        face_img = ImageAlign(image,source_points)
        

        ss = SingleFaceRecog(net,face_img,vector_dict)
        if ss == "unknown":
            randnum = random.randint(1,1000)
            t = int(time.time())
            cv2.imwrite("./pic/unknown/" + str(t) + "_" + str(randnum) + ".jpg",face_img)
        output_result.append(ss)
    
    return output_result



def DrawImage(img_raw,recog_result,detect_result):
    result_image = img_raw
    for k in range(detect_result.shape[0]):
        b = list(map(int, detect_result[k]))
        cv2.rectangle(result_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.circle(img_raw, (b[4 + 0], b[4 + 1]), 2, (255, 0, 0), 2)
        cv2.circle(img_raw, (b[4 + 2], b[4 + 3]), 2, (0, 0, 255), 2)
        cv2.circle(img_raw, (b[4 + 4], b[4 + 5]), 2, (0, 255, 255), 2)
        cv2.circle(img_raw, (b[4 + 6], b[4 + 7]), 2, (255, 255, 0), 2)
        cv2.circle(img_raw, (b[4 + 8], b[4 + 9]), 2, (0, 255, 0), 2)
        #cv2.putText(result_image, "Peter " + str(detect_result[k][14]), (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX,0.7,(0, 0, 255))
        cv2.putText(result_image, recog_result[k], (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX,0.7,(0, 0, 255))
    return result_image


def updateVector(detect_model,recog_model):
    output_result = {}
    total = []
    for folder in os.listdir("./pic"):
        if len(os.listdir("./pic/" + folder)) ==0:
            continue
        if(folder == "unknown"):
            continue
        
        vectors = []
        for name in os.listdir("./pic/" + folder):
            img_raw = cv2.imread("./pic/" + folder +"/"+name)
            #if(img_raw.shape[0] == 0):
            #continue
            #detect_result = FaceDetect(detect_model,img_raw)
            #for result in detect_result[:1]:
            #source_points = result[4:14]
            #face_img = ImageAlign(img_raw,source_points)
            vector = Face2Vector(recog_model,img_raw)
            vectors.append(vector)
        

        output_result[folder] = np.mean(vectors,axis=0)
        total.append(output_result[folder])
    
    #print(output_result)
    output_result["mu"] = np.mean(total,axis=0)
    print(len(output_result["mu"]))

    for key in output_result.keys():
        if (key == "mu"):
            continue
        vector = output_result[key] - output_result["mu"]
        vector = vector/np.sqrt(np.sum(np.power(vector, 2)))

        output_result[key] = vector

    np.save(config.recog_vector_path, output_result)
    #exit(0)


if __name__ == "__main__":
    from model.shufflenet import ShuffleFaceNet
    from model.yufacedetectnet import YuFaceDetectNet
    import torch
    import torch.backends.cudnn as cudnn
    #recognition model init
    device = torch.device(config.device)
    torch.set_grad_enabled(False)
    cudnn.benchmark = True

    #detect model init
    detect_model = YuFaceDetectNet(phase='test', size=None )
    detect_model = load_model(detect_model, config.detect_model_path, True)
    detect_model.eval()
    detect_model = detect_model.to(device)

    #recognition model init
    recog_model = ShuffleFaceNet()
    recog_model = load_model(recog_model,config.recog_model_path,True)
    recog_model.eval()
    updateVector(detect_model,recog_model)


    