import sys
import torch
import argparse
from pathlib import Path

import torch
FILE = Path(__file__).resolve()
from alficore.wrapper.test_error_models_imgclass import TestErrorModels_ImgClass
from alficore.dataloader.objdet_baseClasses.common import pytorchFI_objDet_inputcheck, resize
from alficore.ptfiwrap_utils.build_native_model import build_native_model
from typing import Dict, List
from alficore.ptfiwrap_utils.helper_functions import TEM_Dataloader_attr
import torch
import torch.nn as nn

import torch.nn.utils.prune as prune
from captum.attr import NeuronGradient,LayerConductance
from alficore.dataloader.mnist_loader import MNIST_dataloader,CIFAR10_dataloader

from adapt.approx_layers import axx_layers as approxNN
axx_mult = 'mul8s_1L2H'



class LeNet_orig(nn.Module):

    def __init__(self, color_channels=3):
        super(LeNet_orig, self).__init__()

        # Config
        self.ImageSize = (32, 32)
        self.InChannels = color_channels
        # self.Nr_rangers = 7
        # self.Bounds = np.reshape([None] * (self.Nr_rangers * 2), (self.Nr_rangers, 2))
        # if bounds is not None and len(bounds) >= self.Nr_rangers and len(bounds[0]) >= 2:
        #     self.Bounds = np.array(bounds)
        #self.quant = torch.quantization.QuantStub()
        # Layers
        self.convBlock1 = self.make_conv_block(self.InChannels, 6, 5, 0) #Ranger 0,1
        self.convBlock2 = self.make_conv_block(6, 16, 5, 2) #Ranger 2,3
        # self.flatten_Ranger = Ranger(self.Bounds[4]) # Ranger 4
        self.fc1 = self.make_fcc_block(16 * 5 * 5, 120, 5) # Ranger 5
        self.fc2 = self.make_fcc_block(120, 84, 6)  # Ranger 6
        self.fc3 = nn.Linear(84, 10)
        #self.fc3 = approxNN.AdaPT_Linear(84, 10, axx_mult = axx_mult)
        #self.dequant = torch.quantization.DeQuantStub()



    def make_conv_block(self, in_channels, out_channels, kernel_size, ranger_start_nr):
        """
        Creates one convolutional block. Contains two Ranger layers.
        :param in_channels: nr of input channels
        :param out_channels: output channels for conv layer
        :param kernel_size: for conv layer
        :param ranger_start_nr: list in bounds list that the first Ranger layer gets
        :return: Container with convolutional block.
        """
        # Note: Conv2d: default stride = 1, default padding = 0
        # Note: MaxPool2d: default stride = kernel_size, padding = 0
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)]
        layers += [nn.ReLU(inplace=True)]
        # layers += [Ranger(self.Bounds[ranger_start_nr])]
        layers += [nn.MaxPool2d(kernel_size=2)] #stride is by default = kernel size
        # layers += [Ranger(self.Bounds[ranger_start_nr + 1])]  # stride is by default = kernel size

        return nn.Sequential(*layers)


    def make_fcc_block(self, in_channels, out_channels, ranger_start_nr):
        """
        Creates one fcc block. Contains one Ranger layer.
        :param in_channels: fcc input
        :param out_channels: fcc output
        :param ranger_start_nr: list in bounds list that the first Ranger layer gets
        :return: Container with fcc block.
        """
        layers = []
        layers += [nn.Linear(in_channels, out_channels)]
        layers += [nn.ReLU(inplace=True)]
        # layers += [Ranger(self.Bounds[ranger_start_nr])]

        return nn.Sequential(*layers)



    def forward(self, x):
        #x = self.quant(x)
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.dequant(x)
        return x


class  build_objdet_native_model_img_cls(build_native_model):
    """
    Args:
        original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

    Returns:
        predictions (dict):
            the output of the model for one image only.
            See :doc:`/tutorials/models` for details about the format.
    """
    def __init__(self, model, device):
        super().__init__(model=model, device=device)
        ### img_size, preprocess and postprocess can also be inialised using kwargs which will be set in base class
        self.preprocess = True
        self.postprocess = False
        self.model_name = model._get_name().lower()
        if "lenet" in self.model_name:
            self.img_size = 32
        elif "alex" in self.model_name:
            self.img_size = 256
        else:
            self.img_size = 416


    def preprocess_input(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        ## pytorchfiWrapper_Obj_Det dataloaders throws data in the form of list.
        [dict_img1{}, dict_img2(), dict_img3()] -> dict_img1 = {'image':image, 'image_id':id, 'height':height, 'width':width ...}
        This is converted into a tensor batch as expected by the model
        """
        images = [resize(x['image'], self.img_size) for x in batched_inputs]


        # Convert to tensor
        images = torch.stack(images).to(self.device)
        return images

    def postprocess_output(self):
        return

    def __getattr__(self, method):
        if method.startswith('__'):
            raise AttributeError(method)
        try:
        # if hasattr(self.model, method):
            
            try:
                func = getattr(self.model.model, method)
            except:
                func = getattr(self.model, method)
            ## running pytorch model (self.model) inbuilt functions like eval, to(device)..etc
            ## assuming the executed method is not changing the model but rather
            ## operates on the execution level of pytorch model.
            def wrapper(*args, **kwargs):
                if (method=='to'):
                    return self
                else:
                    return  func(*args, **kwargs)
            return wrapper
        except KeyError:
            raise AttributeError(method)

    def __call__(self, input, dummy=False):
        input = pytorchFI_objDet_inputcheck(input, dummy=dummy)

        _input = input
        if self.preprocess:
            _input = self.preprocess_input(input)
        output = self.model(_input)
        
        ####
        #neuron_ig=LayerConductance(self.model,self.model.convBlock1)
        #input = torch.randn(2, 1, 32, 32, requires_grad=True)
        #input=dl_attr.dataloader.images
        #attribution = neuron_ig.attribute(_input, target=1)
        #print(attribution)
        #####
        return output

def main(argv):

    opt = parse_opt()
    
    device = torch.device(
        "cuda:{}".format(opt.device) if torch.cuda.is_available() else "cpu")
    print(device) 
    device = torch.device('cpu')

        ## set dataloader attributes
    dl_attr = TEM_Dataloader_attr()
    dl_attr.dl_random_sample  = opt.random_sample
    dl_attr.dl_batch_size     = opt.dl_batchsize
    dl_attr.dl_shuffle        = opt.shuffle
    dl_attr.dl_sampleN        = opt.sample_size
    dl_attr.dl_num_workers    = opt.num_workers
    dl_attr.dl_device         = device
    dl_attr.dl_dataset_name   = opt.dl_ds_name
    dl_attr.dl_img_root       = opt.dl_img_root
    dl_attr.dl_gt_json        = opt.dl_json

    dataloader = MNIST_dataloader(dl_attr)
    dataloader = CIFAR10_dataloader(dl_attr)  # change imageC in core.py 
    dataloader= dataloader.data_loader
    # Model   ----------------------------------------------------------
    leNet = LeNet_orig(color_channels=1)
    leNet.load_state_dict(torch.load('demo_img_class_resources/lenet5-mnist.pth')) #load the pretrained weights
    leNet = leNet.to(device)
    #pruning
    #leNet.convBlock1[0]=prune.random_unstructured(leNet.convBlock1[0], name="weight", amount=0.3)
    leNet.eval()
    #model = leNet
    #ALEXNET pretrained
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.classifier[1] = nn.Linear(9216,4096)
    model.classifier[4] = nn.Linear(4096,1024)
    model.classifier[6] = nn.Linear(1024,10)
    import time
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ## python demo_img_classification_custom.py --config-file default_alexnet.yml --dl-ds-name cifar10 
    #model.load_state_dict(torch.load('/home/tawm9/pytorchalfi/data/model_params_ConvNet1.pkl'))
    model.train()
    for epoch in range(50):  # loop over the dataset multiple times
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            #Time
            end_time = time.time()
            time_taken = end_time - start_time

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1))
                print('Time:',time_taken)
                running_loss = 0.0

    print('Finished Training of AlexNet')
    model.eval()
    #Testing Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
    #model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    #model_fp32_fused = torch.quantization.fuse_modules(model, [['convBlock1']])
    #model_fp32_prepared = torch.quantization.prepare(model)
    #model_int8 = torch.quantization.convert(model_fp32_prepared)
    #model=model_int8

    
    iterator = iter(dataloader)
    inputcaptum,_=next(iterator)
    #inputcaptum= torch.Tensor(inputcaptum)
    #neuron_ig=LayerConductance(model,model.convBlock1)  #for lenet 
    neuron_ig=LayerConductance(model,model)  #for alexnet
    print(model)
    #input = torch.randn(2, 1, 32, 32, requires_grad=True)
    #input=dl_attr.dataloader.images
    attribution = neuron_ig.attribute(inputcaptum, target=1)
    print(attribution.shape)
    import numpy as np
    ordered_attribution= np.argmax(np.asarray(attribution.detach()),axis=0)

    print(ordered_attribution.shape)
    print(np.sum(ordered_attribution))
    
    fault_files = opt.fault_files
    wrapped_model = build_objdet_native_model_img_cls(model, device)

    net_Errormodel = TestErrorModels_ImgClass(model=wrapped_model, resil_model=None, resil_name=None, model_name=model._get_name(), config_location=opt.config_file, \
                    ranger_bounds=None, device=device, ranger_detector=False, inf_nan_monitoring=True, disable_FI=False, dl_attr=dl_attr, num_faults=0, fault_file=fault_files, \
                        resume_dir=None, copy_yml_scenario = False )
    net_Errormodel.test_rand_ImgClass_SBFs_inj()



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dl-json', type=str, default='/nwstore/datasets/ImageNet/imagenet_class_index.json', help='path to datasets ground truth json file')
    parser.add_argument('--dl-img-root', type=str, default='/nwstore/datasets/ImageNet/ILSVRC/random20classes_FI', help='path to datasets images')
    parser.add_argument('--dl-ds-name', type=str, default='Mnist', help='dataset short name')
    parser.add_argument('--config-file', type=str, default='default_lenet.yml', help='name of default yml file - inside scenarios folder')
    parser.add_argument('--fault-files', type=str, default=None, help='directory of already existing fault files to repeat existing experiment')
    parser.add_argument('--dl-batchsize', type=int, default=10, help='dataloader batch size')
    parser.add_argument('--sample-size', type=int, default=100, help='dataloader sample size')
    parser.add_argument('--num-workers', type=int, default=1, help='dataloader number of workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--random-sample', type=bool, default=False, help='randomly sampled of len sample-size from the dataset')
    parser.add_argument('--shuffle', type=bool,  default=False, help='Shuffle the sampled data in dataloader')   
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":

    main(sys.argv)
