
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np


from model_modify import ShuffleFaceNet



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



def Face2Vector(model,image):
    img1 = image
    imglist = [img1]
    for i in range(len(imglist)):
        imglist[i] = (imglist[i] - 127.5) / 128.0
        imglist[i] = imglist[i].transpose(2, 0, 1)
    imgs = [torch.from_numpy(i).float().cpu().unsqueeze(0) for i in imglist]
    vector1 = model(imgs[0]).detach().numpy()[0]

    return vector1

if __name__ == '__main__':
    device = torch.device('cpu:0')
    torch.set_grad_enabled(False)
    cudnn.benchmark = True

    #recognition model init
    recog_model = ShuffleFaceNet()
    recog_model = load_model(recog_model,"300.pth",True)
    recog_model.eval()
    recog_model = recog_model.to(device)


    image = cv2.imread("face.jpg")
    vector = Face2Vector(recog_model,image)

    print(vector)