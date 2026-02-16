# cassava-model-comparison
Cassava Leaf Disease Classification을 기반으로 Hydra 설정을 통해 다양한 실험(exp)을 구성하고 여러 CNN 모델(SimpleCNN, MobileNetV2, ResNet18)의 학습 파이프라인을 관리하기 위한 프로젝트입니다.  
학습 파이프라인은 재현성과 유지보수성을 고려하여 datasets / engine / models 구조로 분리하였으며, Hydra 기반 설정 관리 시스템을 적용했습니다.

## Hydra 기반 설정 관리
실험 설정은 Hydra를 사용하여 YAML 파일로 분리 관리합니다.

- dataset / model / train / exp 설정을 개별 파일로 구성
- CLI override로 다양한 실험 실행 가능
- 코드 수정 없이 실험 조건 변경 가능
```bash
python scripts/train.py model=resnet18 exp=freeze
```
runs 폴더에는 각 실험별 결과가 저장됩니다.

## Dataset Setup
```bash
kaggle competitions download -c cassava-leaf-disease-classification  
Expand-Archive cassava-leaf-disease-classification.zip -DestinationPath data
```

## Train (scripts/train.py)  
모델 학습을 실행합니다.  
각 runs 폴더에는 validation accuracy 기준으로 가장 성능이 높은 모델과 학습 기록이 저장됩니다.
```bash
best.pt
history.pt
```

## Test (scripts/test.py)  
저장된 best 모델을 로드한 뒤 test_loader를 사용하여 최종 테스트 성능을 평가합니다.

## Visualize (scripts/visualize.py)  
학습 과정에서 저장된 history를 기반으로
- Train / Validation Loss
- Train / Validation Accuracy
- Best Validation Accuracy 그래프를 출력합니다. 

여러 모델(SimpleCNN, MobileNetV2, ResNet18)의 Validation 성능 곡선 비교도 가능합니다.

## Engineering Design
여러 모델을 동일한 조건에서 비교하기 위해 공통 학습 루프를 중심으로 구조를 설계했습니다.  
- 모델 생성은 build_model()에서 관리
- training components는 factory에서 생성
- train/evaluate loop 분리

datasets / engine / models 역할을 나누어 코드 가독성과 유지보수성을 고려했습니다.  
Hydra를 사용하여 실험 설정을 코드에서 분리하고, 재현 가능한 실험 구조를 구성했습니다.

## Models
현재 구현된 모델:
- SimpleCNN
- MobileNetV2
- ResNet18

새로운 모델을 추가할 경우, model_factory에만 등록하면 학습 코드 수정 없이 확장 가능합니다.

## Project Structure  
```text
cassava-model-comparison
├─ configs
│  ├─ dataset
│  │  └─ cassava.yaml
│  ├─ exp
│  │  ├─ base.yaml
│  │  └─ freeze.yaml
│  ├─ model
│  │  ├─ mobilenet_v2.yaml
│  │  ├─ resnet18.yaml
│  │  └─ simple_cnn.yaml
│  ├─ paths
│  │  └─ paths.yaml
│  ├─ train
│  │  └─ train.yaml
│  └─ config.yaml
│
├─ data
├─ runs
│
├─ scripts
│  ├─ predict.py
│  ├─ test.py
│  ├─ train.py
│  └─ visualize.py
│
├─ src
│  └─ cassava_model_comparison
│     ├─ datasets
│     │  ├─ __init__.py
│     │  ├─ cassava_dataset.py
│     │  ├─ dataloaders.py
│     │  ├─ split.py
│     │  └─ transforms.py
│     │
│     ├─ engine
│     │  ├─ checkpoint
│     │  │  ├─ __init__.py
│     │  │  ├─ load.py
│     │  │  └─ save.py
│     │  │
│     │  ├─ factories
│     │  │  ├─ __init__.py
│     │  │  └─ training_factory.py
│     │  │
│     │  ├─ loops
│     │  │  ├─ __init__.py
│     │  │  ├─ evaluate_one_epoch.py
│     │  │  └─ train_one_epoch.py
│     │  │
│     │  ├─ __init__.py
│     │  └─ trainer.py
│     │
│     ├─ models
│     │  ├─ __init__.py
│     │  ├─ mobilenet_v2.py
│     │  ├─ model_factory.py
│     │  ├─ resnet18.py
│     │  └─ simple_cnn.py
│     │
│     ├─ utils
│     │  ├─ __init__.py
│     │  └─ image_utils.py
│     │
│     ├─ visualize
│     │  ├─ __init__.py
│     │  └─ plots.py
│     │
│     └─ __init__.py
│
├─ LICENSE
├─ pyproject.toml
├─ README.md
└─ .gitignore
```