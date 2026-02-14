# cassava-model-comparison
PyTorch 기반 Computer Vision 프로젝트로,
Cassava Leaf Disease Classification 문제를 해결하기 위해
여러 CNN 모델(SimpleCNN, MobileNetV2, ResNet18)을 동일한 학습 파이프라인에서 비교했습니다.

본 프로젝트는 단순 모델 학습을 넘어,
구조 설계와 재현 가능한 학습 파이프라인 구축에 초점을 두었습니다.

## Dataset Setup
```bash
kaggle competitions download -c cassava-leaf-disease-classification  
Expand-Archive cassava-leaf-disease-classification.zip -DestinationPath data
```
## Models  
모델은 models/build.py 의 build_model() 함수를 통해 생성됩니다.

## Train (scripts/train.py)  
훈련이 실행되며 각 runs 폴더에는 val_acc(검증 정확도)가 가장 높은 모델의 설정과 기록들이 저장됩니다.  
```bash
best.pt
history.pt
```

## Test (scripts/test.py)  
val_acc(검증 정확도) 기준으로 저장된 best 모델을 로드한 뒤, test_loader를 사용해 최종 테스트 성능을 평가합니다.

## Visualize (scripts/visualize.py)  
디폴트 값으로 SimpleCNN 모델의  
Train / Validation Loss, Train / Validation Accuracy, Best Validation  
그리고 세 모델(SimpleCNN, MobileNetV2, ResNet18)의 Validation Loss / Validation Accuracy 곡선이 출력됩니다.

## Engineering Design
여러 모델을 같은 조건에서 비교하기 위해 공통 학습 루프(fit)를 중심으로 구조를 구성했습니다.  
모델 생성은 build_model() 함수에서 관리하여 새로운 모델을 추가하더라도 학습 코드를 수정하지 않도록 설계했습니다.  
datasets / engine / models 로 역할을 분리해 코드 가독성과 유지보수성을 고려했습니다.

## Project Structure  
```text
cassava-model-comparison
├── data/
├── runs/
│   ├── mobilenet_v2/
│   ├── resnet18/
│   └── simple_cnn/
├── scripts/
│   ├── train.py
│   ├── test.py
│   ├── predict.py
│   └── visualize.py
└── src/
    └── cassava_model_comparison/
        ├── config.py
        ├── datasets/
        │   ├── dataloader.py
        │   ├── dataset.py
        │   ├── split.py
        │   └── transforms.py
        ├── engine/
        │   ├── train.py
        │   ├── eval.py
        │   └── save.py
        ├── io/
        │   └── image_utils.py
        ├── models/
        │   ├── build.py
        │   ├── simple_cnn.py
        │   ├── mobilenet_v2.py
        │   └── resnet18.py
        └── visualize/
            └── plots.py
```