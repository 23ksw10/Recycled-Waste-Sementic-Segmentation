# Recycled-Waste-Sementic-Segmentation

잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립, 소각되기 때문입니다. 우리나라의 분리 수거율은 굉장히 높은 것으로 알려져 있고, 또 최근 이러한 쓰레기 문제가 주목받으며 더욱 많은 사람이 분리수거에 동참하려 하고 있습니다. 하지만 '이 쓰레기가 어디에 속하는지', '어떤 것들을 분리해서 버리는 것이 맞는지' 등 정확한 분리수거 방법을 알기 어렵다는 문제점이 있습니다.

따라서, 우리는 쓰레기가 찍힌 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기로 분류할 것 입니다.

### Usage

```
data
    data
      ├── batch_01_vt
      │   ├── 데이터셋 (이미지)
      │   ├── ...
      │   └──
      ├── batch_02_vt
      │   ├── 데이터셋 (이미지)
      │   ├── ...
      │   └──
      ├── batch_03
      │   ├── 데이터셋 (이미지)
      │   ├── ...
      │   └──
      ├── train_all.json
      ├── train.json
      ├── val.json
      ├── train_data0.json
      ├── ...
      ├── valid_data4.json
      ├── train_data_pesudo0.json
      ├── ...
      ├── valid_data_pesudo4.json
      └── test.json
code
  ├── DeepLabV3Plus
  ├── FPN
  ├── ensemble.py
 

```

### Models

I used [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) (SMP) as a framework for all of my models. 

I used an ensemble of models for my submissions

### Encoders

I used `EfficientNet` to the above framework and had great results. 

### Decoders

I used  `DeepLabV3Plus`  + `FPN` from SMP. 

I alos tried  `Unet` but didn't work well.

### Augmentations

I used the following [Albumentations](https://github.com/albu/albumentations):

```python
A.OneOf([A.CLAHE(),
         A.IAASharpen(alpha=(0.2, 0.3)),
         A.GaussianBlur(3, p=0.3)]
         ,p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
        A.Resize(512,512),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(),
        # A.RandomResizedCrop(512,512,scale = (0.5,0.8)),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2
```

I found these augs worked better than simple crops/flips.

(mIou score : 0.59 → 0.61)

### Loss

I used `(0.6 * BCE) + (0.4 * (1 - Dice))`.

(mIou score increase about 0.2~0.3)

### LR Schedule

I used ReduceLROnPlateau

### Ensembles

 

![Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/media_images_predict_15_14_0.png](Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/media_images_predict_15_14_0.png)

            Original image

![Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/index.png](Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/index.png)

            FPN

![Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/Screenshot_2021-06-03_Weights_Biases.png](Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/Screenshot_2021-06-03_Weights_Biases.png)

           DeepLabV3Plus

![Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/media_images_predict_19_18_68.png](Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/media_images_predict_19_18_68.png)

            Original image
            
![Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/index1.png](Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/index1.png)

            FPN                                              

![Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/Screenshot_2021-06-03_Weights_Biases(1).png](Recycled-Waste-Sementic-Segmentation%20f459b5f17b4e469a83dc3a08087c39ef/Screenshot_2021-06-03_Weights_Biases(1).png)

           DeepLabV3Plus


As you can see FPN and DeepLabV3Plus model predict different features and shape.

So i decided ensemble both models
