# COCO 데이터 준비

[COCO 공식 다운로드](https://cocodataset.org/#download)에서 2017 train·validation 이미지와 captions annotation을 준비합니다.
데이터는 제공처의 이용 조건을 따릅니다.

```text
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

학습은 첫 번째 캡션을 사용하고, 이미지는 정사각형 resize 후 텐서로 변환합니다.
텍스트 길이는 기본 128개 토큰으로 제한합니다. 이미지 크기는 16의 배수로 지정합니다.
기본값은 256이며 VAE latent의 한 변은 32입니다.
