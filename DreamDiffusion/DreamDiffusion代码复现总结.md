# Reproduce_DreamDiffusion

## 数据准备

#### 根目录 `eegdataset` 下：

- 数据文件
  - `eeg_signals_raw_with_mean_std.pth`
  - `eeg_55_95_std.pth`
  - `eeg_14_70_std.pth`
  - `eeg_5_95_std.pth`
  - `block_splits_by_image_singl.pth`
  - `block_splits_by_image_all.pth`
- 模型文件
  - `v1-5-pruned.ckpt`
  - `checkpoint_best.pth`
- 图片文件
  - 目录 `imageNet_images` 下放置ImageNet图片的一个子集。

#### 模型目录`gm_pretrain`下（对应`pretrains`）

查看下方。

### 数据预处理 Data Preprocess

- 将eeg的 `.pth` 文件放在根目录 `eegdataset` 下，依次运行至 `Now save all the files - don't run this unless eeg is not populated` 得到文件夹 `subject` \ `label` \ `image` \ `eeg`，每个单独的eeg的`.npy`文件存放在 `eeg` 中。
- 注意根路径为 `/content/drive/My Drive/eegdataset/` ,其下文件夹格式除了上述转换好的数据文件夹之外，其他文件夹结构与代码[bbaaii/DreamDiffusion: Implementation of “DreamDiffusion: Generating High-Quality Images from Brain EEG Signals” (github.com)](https://github.com/bbaaii/DreamDiffusion) 相同。预训练好的模型都放在`pretrains`文件夹下
- `pretrains`
  - `models`
    - `config.yaml` 文件，注意其中的每个 `target` 类应指向代码中的对应类，需要作出修改。
    - `v1-5-pruned.ckpt` 原始的`v1.5 diffusion` 模型
  - `generation` (fine-tuning stable diffusion得到的文件)
    - `checkpoint_best.pth`
  - `eeg_pretrain`  （EEG pre-training得到的文件）
    - `checkpoint.pth  (pre-trained EEG encoder)`

## 训练EEG pretrained

运行 `Pretrain EEG state` 即可。注意`config.yaml`文件中的每个 `target` 类应指向代码中的对应类。

通过

```python
checkpoint_path = '/content/drive/MyDrive/eegdataset/results/eeg_pretrain/26-10-2023-11-18-06/checkpoints/checkpoint.pth'
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")
```

可以加载某个训练好的checkpoint文件，从其中的某个epoch开始继续训练。

## 用预训练好的EEG表示（VAE Encoder）和已有的生成模型（Diffusion）生成图片

#### 代码中需要修改至自己的文件的部分

```python
# sd文件指向fine tuning好的stable diffusion的checkpoint文件。
sd = torch.load('/content/drive/MyDrive/eegdataset/checkpoint.pth', map_location='cpu')
# 根目录
config.root_path = '/content/drive/MyDrive/eegdataset/'
# eeg数据预训练好的encoder，在上述训练步骤中训练
config.pretrain_mbm_path = '/content/drive/MyDrive/eegdataset/results/eeg_pretrain/26-10-2023-16-30-54/checkpoints/checkpoint.pth'
# gm_pretrain文件夹
config.pretrain_gm_path = '/content/drive/MyDrive/eegdataset/gm_pretrain'
```

#### Image图片文件的使用情况：

在eeg数据中，作者为了构建数据集，将eeg数据中对应得给了image的标签，image标签用的就是imagenet里的数据图片。所以在生成过程中，Dataloading中的getitem也要去根据traindata、testdata（eeg数据文件）中先去得到对应的图片image标签。





