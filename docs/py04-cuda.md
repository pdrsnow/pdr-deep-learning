# 安装`CUDA`

## 1. 安装显卡驱动

> - 如果已安装过显卡驱动，可通过`nvidia-smi`指令查看支持的最高`CUDA`版本，直接去安装该版本的`CUDA`；
> - 查找`CUDA`版本与驱动版本的关系: <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4>
> - 这里样例`CUDA Version: 12.0`，结合当时机器学习主流版本选择`11.8`版本

```
PS C:\Users\YouName> nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 526.47       Driver Version: 526.47       CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
|  0%   52C    P0    82W / 290W |   1611MiB /  8192MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

> + SD(Studio Drivers): 稳定性优先，适用于工作环境，驱动文件以`nsd-dch-whql.exe`结尾
> + GRD(Game Ready Drivers): 新特性(patches, DLC ets)， 使用于娱乐，驱动文件以`dch-whql.exe`结尾

```
# 显卡驱动下载(自行选择显卡型号和系统版本)
https://www.nvidia.com/download/index.aspx

# 下面是驱动文件样例, 最前面的`555.85`为驱动版本号
#555.85-desktop-win10-win11-64bit-international-nsd-dch-whql.exe
#555.85-desktop-win10-win11-64bit-international-dch-whql.exe
```

## 2. 下载安装`CUDA`

> + 最新cuda下载地址: <https://developer.nvidia.com/cuda-downloads>
> + 历史cuda下载地址(建议): <https://developer.nvidia.com/cuda-toolkit-archive>

```shell
# Windwos -> x86_64 -> 11 -> exe(local)
# https://developer.nvidia.com/cuda-11-8-0-download-archive

# cuda_11.8.0_522.06_windows.exe
curl -OL 'https://developer.download.nvidia.cn/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe'
```

> 使用`nvcc --version`检查`CUDA`安装状态

```
PS C:\Users\YouName> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

## 3. 下载安装`cuDNN`

> + 下载地址: <https://developer.nvidia.com/cudnn-downloads>
> + 历史下载: <https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/>

```shell
# cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip
curl -OL "https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn/\
windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip"

# 下载后解压到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
```

```
cd /d "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
mklink /H cudart64_110.dll  cudart64_12.dll
mklink /H cublas64_11.dll   cublas64_12.dll
mklink /H cublasLt64_11.dll cublasLt64_12.dll
mklink /H cufft64_10.dll    cufft64_11.dll
mklink /H cusparse64_11.dll cusparse64_12.dll

为 cudart64_110.dll <<===>> cudart64_12.dll 创建了硬链接
为 cublas64_11.dll <<===>> cublas64_12.dll 创建了硬链接
为 cublasLt64_11.dll <<===>> cublasLt64_12.dll 创建了硬链接
为 cufft64_10.dll <<===>> cufft64_11.dll 创建了硬链接
为 cusparse64_11.dll <<===>> cusparse64_12.dll 创建了硬链接
```
