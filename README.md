# cassava-model-comparison
Comparing SimpleCNN, MobileNetV2, and ResNet18 on Cassava Leaf Disease Classification

kaggle competitions download -c cassava-leaf-disease-classification

Windows
Expand-Archive cassava-leaf-disease-classification.zip -DestinationPath data

Linux / Mac
unzip cassava-leaf-disease-classification.zip -d data

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