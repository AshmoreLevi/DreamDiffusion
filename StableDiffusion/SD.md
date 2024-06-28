# Stable Diffusion学习笔记

发展：2020扩散模型DDPM-2022Latent Diffusion Model-Stable Difffusion

- DDPM
  - 参数化的马尔可夫链
  - 前向过程（扩散过程）与反向过程（生成过程）
  - 实际训练过程：训练UNet预测每一步的noise
  - ![image-20240117092239059](/Users/levi/Library/Application Support/typora-user-images/image-20240117092239059.png)
- Latent Diffusion Models（LDMs）即SD的模型整体架构
  - 文献：CVPR 2022《High-Resolution Image Synthesis with Latent Diffusion Models》
  - 代码：[CompVis/latent-diffusion: High-Resolution Image Synthesis with Latent Diffusion Models (github.com)](https://github.com/CompVis/latent-diffusion)
  - <img src="/Users/levi/Library/Application Support/typora-user-images/image-20240117092753477.png" alt="image-20240117092753477" style="zoom:50%;" />
  - 改进：
    - 加入Autoencoder（左侧红色部分），使得扩散过程在**latent space**下，提高图像生成的效率；
    - 加入**条件机制**，能够使用其他模态的数据控制图像的生成（上图中右侧灰色部分），其中条件生成控制通过Attention（上图中间部分QKV）机制实现。
    - LDM中  是从 encoder 获取到的低维表达，LDM的UNet只需要预测低维空间噪声

## 核心网络结构

主要由VAE，U-Net以及CLIP Text Encoder三个核心组件构成。

FP16精度下Stable Diffusion模型大小2G（FP32：4G），其中U-Net大小1.6G，VAE模型大小160M以及CLIP Text Encoder模型大小235M。

### VAE

**在Stable Diffusion中，VAE模型主要起到了图像压缩和图像重建的作用**。

- VAE的Encoder（编码器）结构能将输入图像转换为低维Latent特征，并作为U-Net的输入
- VAE的Decoder（解码器）结构能将低维Latent特征重建还原成像素级图像
- 具体结构：![image-20240117103016566](/Users/levi/Library/Application Support/typora-user-images/image-20240117103016566.png)
- 训练过程与损失函数
  - **L1回归损失**
  - **感知损失**：比较原始图像和生成图像在传统深度学习模型不同层中的特征图之间的相似度，而不直接进行像素级别的对比。
  - **基于patch的对抗训练策略**：判别器架构不再评估整个生成图像是否真实，而是评估生成图像中的patch（局部区域）是否真实。

### **CLIP Text Encoder模型**

**文本编码模块直接决定了语义信息的优良程度。**CLIP（Contrastive Language-Image Pre-Training）模型是一个基于对比学习的**多模态模型**，主要包含Text Encoder和Image Encoder两个模型。

CLIP模型通过将标签文本和图片提取embedding向量，然后用**余弦相似度（cosine similarity）**来比较两个embedding向量的**相似性**

- Text Encoder
- Image Encoder

所提取的文本特征Context通过CrossAttention嵌入Unet中，即文本特征作为Attention的key、value；Unet的特征作为query。

![image-20240117141934362](/Users/levi/Library/Application Support/typora-user-images/image-20240117141934362.png)

### UNet 

![img](https://img-blog.csdnimg.cn/0d62acfc293f4424ba2b3ce20de0e122.png#pic_center)

**预测噪声残差**，并结合 Sampling method 对输入的特征矩阵进行重构，**逐步将其从随机高斯噪声转化成图片的Latent Feature**。

- 【待check】前向推理（difussion process）：将预测出的噪声残差从原噪声中去除，得到逐步去噪后的latent feature。
- 生成重建：通过**VAE的Decoder**将Latent Feature重建成像素级图像。

SD中的U-Net，在传统深度学习时代的Encoder-Decoder结构的基础上，增加了：

- **ResNetBlock（包含Time Embedding）模块**：time embedding引入时间信息**时间步长T**，同时告诉U-Net处在迭代过程的哪一步。
  - 输入：Latent Feature、Time Embedding+GSC
  - 输出：
- **Spatial Transformer**（SelfAttention + CrossAttention + FeedForward）模块
  - **CrossAttention**
    - **在图片对应的位置上融合了语义信息，是将本文与图像结合的一个模块**
    - 输入：ResNetBlock模块的输出、CLIP后的Context Embedding
  - BasicTransformer Block（**SelfAttention** + CrossAttention + FeedForward）
  - Spatial Transformer（+GroupNorm和两个卷积层）
- CrossAttnDownBlock，CrossAttnUpBlock和CrossAttnMidBlock模块
  - U-Net的Encoder部分中，**使用了三个CrossAttnDownBlock模块**
  - U-Net的Decoder部分中，**使用了三个CrossAttnUpBlock模块**
  - CrossAttnMidBlock模块中，包含ResNetBlock Structure+BasicTransformer Block+ResNetBlock Structure
  - 即**ResNetBlock Structure+BasicTransformer Block+Downsample/Upsample/ResNetBlock**
- ![image-20240117142124220](/Users/levi/Library/Application Support/typora-user-images/image-20240117142124220.png)

## 代码

仓库：[CompVis/latent-diffusion: High-Resolution Image Synthesis with Latent Diffusion Models (github.com)](https://github.com/CompVis/latent-diffusion)

### 项目结构

<img src="/Users/levi/Library/Application Support/typora-user-images/image-20240119133526321.png" alt="image-20240119133526321" style="zoom:50%;" />

- models（模型的下载）
  - first_stage_models
  - ldm
- scripts
- ldm
  - data
  - models（模型）
    - diffusion
    - autoencoder.py
  - modules（模块）
    - **diffusionmodules**
      - model.py
      - openaimodel.py
        - Class ResBlock
        - Class Upsample
        - Class Downsample
        - Class AttentionBlock
        - Class UnetModel
    - distribution
    - **encoders**
    - image_degradation
    - losses
    - **attention.py**：crossAttention
      - Class SpatialSelfAttention
      - Class CrossAttention
      - Class BasicTransformerBlock
      - Class SpatialTransformer
    - x_transformer.py
    - ema.py



