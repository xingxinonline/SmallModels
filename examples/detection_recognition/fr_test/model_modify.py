import torch
import torch.nn as nn
import math
import numpy as np
import sys
import os
import cv2

sys.path.append(os.getcwd() + '../')

import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

# http://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Martindez-Diaz_ShuffleFaceNet_A_Lightweight_Face_Architecture_for_Efficient_and_Highly-Accurate_Face_ICCVW_2019_paper.pdf

def combine_conv_bn(conv, bn):
    conv_result = nn.Conv2d(conv.in_channels, conv.out_channels, 
                            kernel_size=conv.kernel_size, stride=conv.stride, 
                            padding=conv.padding, groups = conv.groups, bias=True)
    
    scales = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    with torch.no_grad():
        conv_result.bias[:] = ( - bn.running_mean) * scales + bn.bias
    for ch in range(conv.out_channels):
        with torch.no_grad():
            conv_result.weight[ch, :, :, :] = conv.weight[ch, :, :, :] * scales[ch]

    return conv_result

def convert_param2string(conv, name, is_depthwise=False, isfirst3x3x3=False, precision='.3g'):
    '''
    Convert the weights to strings
    '''
    (out_channels, in_channels, width, height) = conv.weight.size()

    if (isfirst3x3x3):
        w = conv.weight.detach().numpy().reshape((-1,27))
        w_zeros = np.zeros((out_channels ,5))
        w = np.hstack((w, w_zeros))
        w = w.reshape(-1)
    elif (is_depthwise):
        w = conv.weight.detach().numpy().reshape((-1,9)).transpose().reshape(-1)
    else:
        w = conv.weight.detach().numpy().reshape(-1)

    b = conv.bias.detach().numpy().reshape(-1)

    if (isfirst3x3x3):
        lengthstr_w = str(out_channels) + '* 32 * 1 * 1'
        # print(conv.in_channels, conv.out_channels, conv.kernel_size)
    else:
        lengthstr_w = str(out_channels) + '*' + str(in_channels) + '*' + str(width) + '*' + str(height)
    resultstr = 'float ' + name + '_weight[' + lengthstr_w + '] = {'

    for idx in range(w.size - 1):
        resultstr += (format(w[idx], precision) + ',')
    resultstr += (format(w[-1], precision))
    resultstr += '};\n'

    resultstr += 'float ' + name + '_bias[' + str(out_channels) + '] = {'
    for idx in range(b.size - 1):
        resultstr += (format(b[idx], precision) + ',')
    resultstr += (format(b[-1], precision))
    resultstr += '};\n'

    return resultstr


def convert_param2string_1d(conv, name, is_depthwise=False, isfirst3x3x3=False, precision='.3g'):
    '''
    Convert the weights to strings
    '''
    (out_channels, in_channels, width) = conv.weight.size()


    w = conv.weight.detach().numpy().reshape(-1)

    b = conv.bias.detach().numpy().reshape(-1)


    lengthstr_w = str(out_channels) + '*' + str(in_channels) + '*' + str(width)
    resultstr = 'float ' + name + '_weight[' + lengthstr_w + '] = {'

    for idx in range(w.size - 1):
        resultstr += (format(w[idx], precision) + ',')
    resultstr += (format(w[-1], precision))
    resultstr += '};\n'

    resultstr += 'float ' + name + '_bias[' + str(out_channels) + '] = {'
    for idx in range(b.size - 1):
        resultstr += (format(b[idx], precision) + ',')
    resultstr += (format(b[-1], precision))
    resultstr += '};\n'

    return resultstr

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def create_channel_shuffle_conv_kernel(num_channels, num_groups):
    channels_per_group = num_channels // num_groups
    conv_kernel = torch.zeros(num_channels, num_channels, 1, 1)
    for k in range(num_channels):
        index = (k % num_groups) * channels_per_group + k // num_groups
        conv_kernel[k, index, 0, 0] = 1
    return conv_kernel

def channel_shuffle_v2(x, num_groups):

    batch_size, num_channels, height, width = x.size()
    assert num_channels % num_groups == 0

    conv_kernel = create_channel_shuffle_conv_kernel(num_channels, num_groups)

    return torch.conv2d(x, conv_kernel)

class InvertedResidual1(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.PReLU(),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.PReLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.PReLU(),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle_v2(out, 2)

        return out
    
    def convert_to_cppstring(self, varname):
        rs1 = convert_param2string(self.conv1, varname+'_1', False)
        if self.withBNRelu:
            rs2 = convert_param2string(combine_conv_bn(self.conv2, self.bn), varname+'_2', True)
        else:
            rs2 = convert_param2string(self.conv2, varname+'_2', True)

        return rs1 + rs2

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride,index):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        self.index = index
        self.dp_conv1 = self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1)
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(branch_features)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.dp_conv2 = self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1)
        self.conv3 = nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.dp_conv1,
                self.bn1,
                self.conv1,
                self.bn2,
                self.prelu,
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            self.conv2,
            self.bn2,
            self.prelu,
            self.dp_conv2,
            self.bn2,
            self.conv3,
            self.bn2,
            self.prelu,
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle_v2(out, 2)

        return out
    
    def convert_to_cppstring(self, varname):
        result = ""
        HEAD = []


        #stride 有可能不是1或者大于1么
        if self.stride == 1:
            result += convert_param2string(combine_conv_bn(self.conv2, self.bn2), varname+'_'+str(self.stride)+'_1', False)
            HEAD.append([self.conv2.in_channels, self.conv2.out_channels, 'false', 'true','true', varname+'_'+str(self.stride)+'_1'])
            result += convert_param2string(combine_conv_bn(self.dp_conv2, self.bn2), varname+'_'+str(self.stride)+'_2', True)
            HEAD.append([self.dp_conv2.in_channels, self.dp_conv2.out_channels, 'true', 'false','false', varname+'_'+str(self.stride)+'_2'])
            result += convert_param2string(combine_conv_bn(self.conv3, self.bn2), varname+'_'+str(self.stride)+'_3', False)
            HEAD.append([self.conv3.in_channels, self.conv3.out_channels, 'false', 'true','true', varname+'_'+str(self.stride)+'_3'])
        elif self.stride > 1:
            result += convert_param2string(combine_conv_bn(self.dp_conv1, self.bn1), varname+'_'+str(self.stride)+'_4', True)
            HEAD.append([self.dp_conv1.in_channels, self.dp_conv1.out_channels, 'true', 'false','false', varname+'_'+str(self.stride)+'_4'])
            result += convert_param2string(combine_conv_bn(self.conv1, self.bn2), varname+'_'+str(self.stride)+'_5', False)
            HEAD.append([self.conv1.in_channels, self.conv1.out_channels, 'false', 'true','true', varname+'_'+str(self.stride)+'_5'])
            result += convert_param2string(combine_conv_bn(self.conv2, self.bn2), varname+'_'+str(self.stride)+'_6', False)
            HEAD.append([self.conv2.in_channels, self.conv2.out_channels, 'false', 'true','true', varname+'_'+str(self.stride)+'_6'])
            result += convert_param2string(combine_conv_bn(self.dp_conv2, self.bn2), varname+'_'+str(self.stride)+'_7', True)
            HEAD.append([self.dp_conv2.in_channels, self.dp_conv2.out_channels, 'false', 'true','true', varname+'_'+str(self.stride)+'_7'])
            result += convert_param2string(combine_conv_bn(self.conv3, self.bn2), varname+'_'+str(self.stride)+'_8', False)
            HEAD.append([self.conv3.in_channels, self.conv3.out_channels, 'false', 'true','true', varname+'_'+str(self.stride)+'_8'])
        


        return result,HEAD

class ShuffleFaceNet(nn.Module):
    def __init__(self, stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024], inverted_residual=InvertedResidual):
        super(ShuffleFaceNet, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1_1 = nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(output_channels)

        self.conv1 = nn.Sequential(
            self.conv1_1,
            self.bn1_1,
            nn.PReLU(),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2, 0)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1, i))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]

        self.conv5_1 = nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False)
        self.bn5_1 = nn.BatchNorm2d(output_channels)

        self.conv5 = nn.Sequential(
            self.conv5_1,
            self.bn5_1,
            nn.PReLU(),
        )
        input_channels = output_channels

        self.gdc_conv_1 = nn.Conv2d(input_channels, output_channels, kernel_size=5, stride=1, padding=0, bias=False, groups=input_channels)
        self.gdc_conv_2 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=0, bias=False, groups=input_channels)
        self.gdc_bn_1 = nn.BatchNorm2d(output_channels)
        self.gdc_bn_2 = nn.BatchNorm2d(output_channels)


        self.gdc = nn.Sequential(
            self.gdc_conv_1,
            self.gdc_bn_1,
            nn.PReLU(),
            self.gdc_conv_2,
            self.gdc_bn_2,
            nn.PReLU(),
        )

        input_channels = output_channels
        output_channels = 128

        self.linearconv = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm2d(output_channels)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        #print(x.size())
        x = nn.functional.interpolate(x, size=[112, 112])

        # print(x[0,0,0,0:10])
        # exit(0)

        x = self.conv1(x)
        # x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        #x = x.mean([2, 3])  # globalpool 
        x = self.gdc(x)
        # x = np.squeeze(x, axis=2)
        x = x.view(x.size(0), 1024, 1)
        x = self.linearconv(x)
        x = x.view(x.size(0), 128, 1, 1)
        x = self.bn(x)

        x = x.view(x.size(0), -1)

        
        return x

    def forward(self, x):
        return self._forward_impl(x)


    def export_cpp(self, filename):
        '''This function can export CPP data file for shuffle face net'''
        result_str = '// Auto generated data file\n'
        result_str += '// Copyright (c) 2022-2024, Yanjun Li, all rights reserved.\n'
        result_str += '#include "shufflefacenet.h" \n\n'

        HEAD = []

        result_str += convert_param2string(combine_conv_bn(self.conv1_1,self.bn1_1),"conv_1")
        head_title = [self.conv1_1.in_channels,
                        self.conv1_1.out_channels,
                        'false', # is_depthwise 
                        'false', # is_pointwise
                        'true', # with_relu
                        'conv_1']
        HEAD.append(head_title)

        for i in range(len(self.stage2)):
            weight_str,title = self.stage2[i].convert_to_cppstring("stage2_" + str(i))
            result_str += weight_str
            HEAD = HEAD + title

        for i in range(len(self.stage3)):
            weight_str,title = self.stage3[i].convert_to_cppstring("stage3_" + str(i))
            result_str += weight_str
            HEAD = HEAD + title

        for i in range(len(self.stage4)):
            weight_str,title = self.stage4[i].convert_to_cppstring("stage4_" + str(i))
            result_str += weight_str
            HEAD = HEAD + title

        result_str += convert_param2string(combine_conv_bn(self.conv5_1,self.bn5_1),"conv_5")
        head_title = [self.conv5_1.in_channels,
                        self.conv5_1.out_channels,
                        'false', # is_depthwise 
                        'false', # is_pointwise
                        'true', # with_prelu
                        'conv_5']
        HEAD.append(head_title)
        result_str += convert_param2string(combine_conv_bn(self.gdc_conv_1,self.gdc_bn_1),"gdc_1")
        head_title = [self.gdc_conv_1.in_channels,
                        self.gdc_conv_1.out_channels,
                        'false', # is_depthwise 
                        'false', # is_pointwise
                        'true', # with_prelu
                        'gdc_1']
        HEAD.append(head_title)
        result_str += convert_param2string(combine_conv_bn(self.gdc_conv_2,self.gdc_bn_2),"gdc_2")
        head_title = [self.gdc_conv_2.in_channels,
                        self.gdc_conv_2.out_channels,
                        'false', # is_depthwise 
                        'false', # is_pointwise
                        'true', # with_prelu
                        'gdc_2']
        HEAD.append(head_title)

        result_str += convert_param2string_1d(self.linearconv,"linearconv")

        head_title = [self.gdc_conv_2.in_channels,
                        self.gdc_conv_2.out_channels,
                        'false', # is_depthwise 
                        'false', # is_pointwise
                        'true', # with_relu
                        'linearconv']
        HEAD.append(head_title)


        # # convert to a string
        num_conv = len(HEAD)

        result_str += 'ConvInfoStruct param_pConvInfo[' + str(num_conv*2 - 1) + '] = { \n'

        print(HEAD)
        
        for idx in range(0, num_conv):
            result_str += ('    {' +
                           str(HEAD[idx][0]) + ', ' +
                           str(HEAD[idx][1]) + ', ' +
                           str(HEAD[idx][2]) + ', ' +
                           str(HEAD[idx][3]) + ', ' +
                           str(HEAD[idx][4]) + ', ' +
                           str(HEAD[idx][5])+'_weight'+ ', ' +
                           str(HEAD[idx][5])+'_bias' +'}')

            result_str += ','
            result_str += '\n'

        result_str += '};\n'
        

        # write the content to a file
        with open(filename, 'w') as f:
            f.write(result_str)
            f.close()

        return 0 


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class CosFace_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, s=32.0, m=0.5):
        super(CosFace_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        
        
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c).cuda()
        

        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))
        

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        
        margin_logits = self.s * (logits - y_onehot)
        




        return logits, margin_logits


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



if __name__ == "__main__":
    recog_model = ShuffleFaceNet()
    recog_model = load_model(recog_model,"300.pth",True)
    recog_model.eval()
    recog_model = recog_model.to("cpu:0")

    img1 = cv2.imread("face.jpg")
    img = (img1 - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    img_input = torch.from_numpy(img).float().cpu().unsqueeze(0)


    vector1 = recog_model(img_input).detach().numpy()[0]
    print(vector1)