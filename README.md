# Saliency Attention-based DETR
> **Nam, Kwangwoon**, et al. "SA-DETR: Saliency Attention-based DETR for salient object detection." *Pattern Analysis and Applications* 28.1 (2025): 1-11.

Paper : https://link.springer.com/article/10.1007/s10044-024-01379-5


# Overall Architecture
![SA-DETR_architecture](https://github.com/Namkwangwoon/Saliency-Attention-based-DETR/assets/19163372/3c340a4b-077e-4fc6-8fc1-ae33c55deb35)


# Prerequisite
- PyTorch >=1.5.0
- Requirements
  ```shell
  pip install -r requirements.txt
  ```


# Dataset
### SOC dataset
> Fan, Deng-Ping, et al. "Salient objects in clutter." IEEE Transactions on Pattern Analysis and Machine Intelligence 45.2 (2022): 2344-2366.
- https://github.com/DengPingFan/SODBenchmark


# Training
```shell
python main_SOC.py \
  --masks \
  --no_aux_loss \
  --output_dir "output_path" \
  --epochs 200 \
  --frozen_weights detr-r50-e632da11.pth (or --frozen_weights detr-r101-2c7b67e5.pth --backbone resnet101)
  [--resume "output_checkpoint_path" --lr "lr" --lr_drop "lr_drop"]
```


# Inference
```shell
python main_SOC.py --masks --no_aux_loss --eval
```


# Evaluation
```shell
python pred_SOC.py --masks --no_aux_loss --eval
```
### MAE
> Perazzi, Federico, et al. "Saliency filters: Contrast based filtering for salient region detection." 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012.

![image](https://github.com/Namkwangwoon/Saliency-Attention-based-DETR/assets/19163372/41e0d600-5974-46cb-a9a4-80f0f951261f)
- $N$ : Pixel numbers
- $Sal$ : Saliency(Output) map
- $G$ : GT map

### S-measure
> Fan, Deng-Ping, et al. "Structure-measure: A new way to evaluate foreground maps." Proceedings of the IEEE international conference on computer vision. 2017.

![image](https://github.com/Namkwangwoon/Saliency-Attention-based-DETR/assets/19163372/4807f5b9-ba20-4010-8637-f4c6512dcd77)
- $\alpha$ : Balanced parameter, [0, 1], (0.5 default)
- $S_o$ : Object-aware structural similarity
- $S_r$ : Region-aware structure similarity

### E-measure
> Fan, Deng-Ping, et al. "Enhanced-alignment measure for binary foreground map evaluation." arXiv preprint arXiv:1805.10421 (2018).

![image](https://github.com/Namkwangwoon/Saliency-Attention-based-DETR/assets/19163372/7fa54b28-05e7-4d66-881f-5a3ba2902606)
- $w, h$ : Width, height of map
- $\phi_{FM}$ : Enhanced alignment matrix of forground map


# Results

<img width="50%" alt="스크린샷 2024-02-01 오후 9 44 51" src="https://github.com/Namkwangwoon/Saliency-Attention-based-DETR/assets/19163372/fa9494b0-af47-4389-b5cb-a94b985128ef">

<img width="100%" alt="스크린샷 2024-02-01 오후 9 46 11" src="https://github.com/Namkwangwoon/Saliency-Attention-based-DETR/assets/19163372/693434ef-a739-456f-8958-29c162dc0c66">


# Ablation Studied
Ablation studies of Saliency Module(SM)
### Objects
![image](https://github.com/Namkwangwoon/Saliency-Attention-based-DETR/assets/19163372/b1febc82-2009-49b3-a842-dd447fb69ba1)
- Without SM, salient objects are not detected, or other objects are detected as salient.

### Attention maps & Object-level masks
![image](https://github.com/Namkwangwoon/Saliency-Attention-based-DETR/assets/19163372/82701137-b8d0-438d-914d-52863b0562ca)
- With SM, each attention map recognizes the shape of an object well, resulting in an accurate object-level mask.


# Reference Codes
- https://github.com/facebookresearch/detr
- https://github.com/DengPingFan/SODBenchmark
- https://github.com/mczhuge/SOCToolbox
