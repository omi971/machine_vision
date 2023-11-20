### How to install Cuda

#### 1. Install NVIDIA Studio Driver
Go for the NVIDIA Studio Driver if you don't
find your graphics cards studio driver then you can
download NVIDIA Game Ready Driver.
- [Download NVIDIA Driver](https://www.nvidia.com/download/index.aspx)

#### 2. CUDA Download Configuration
- architecture: x86_64
- version: 11
- Installer type: exe(local)
- [Download NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

#### 3. cuDNN Download Configuration
- Download cuDNN after installing the CUDA toolkit
- Download cuDNN depending the the version of CUDA toolkit
- [Download cuDNN Library](https://developer.nvidia.com/rdp/cudnn-download)

Here my CUDA toolkit version is 12.3 that is why I have downloaded cuDNN 12.x

#### 4. Download Pytorch
- PyTorch Build: stable
- Package: Pip
- Language: Python
- Compute Platform: CUDA 12.1 (depending on your cuda version)
- [Go to Pytorch site](https://pytorch.org/get-started/locally/)

1. now copy this pip command from the site and then open your pycharm
2. Click on Terminal
3. Now if you are in local terminal
   1. Change it to command prompt
   2. Make sure virtual enviroment is activated
4. Now Paste this command in Command Prompt


Now you're good to go, run a computer vision detection code and make sure in the console 
your external GPU name mentions for my case that is

`Ultralytics YOLOv8.0.26  Python-3.10.9 torch-2.1.1+cu121 CUDA:0 (NVIDIA GeForce MX350, 2048MiB)
YOLOv8n summary (fused): 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs`
 

Here you can see that is can see my GPU Name (**NVIDIA GeForce MX350**) in the output terminal after running the code

