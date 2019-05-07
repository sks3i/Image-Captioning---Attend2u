Image Captioning - Attend2u <br />
---

Implementation of Attend to You: Personalized Image Captioning with Context Sequence Memory Networks <br />

### Requirements
* Python 3.5
* TensorFlow > 1.10
* Other requirements - check requirements.txt


Dataset:
[JSON](https://drive.google.com/uc?export=download&id=0B3xszfcsfVUBdG0tU3BOQWV0a0E) file. <br />
[Images](https://drive.google.com/uc?export=download&id=0B3xszfcsfVUBVkZGU2oxYVl6aDA) <br />

### Running Code

1. Download ResNet trained model
	Run scripts/download_pretrained_resnet_101.sh

2. Extract images features
	Run scripts/extract_features.sh
	
3. Configure the network.
	Edit configs.py
	
4. Train
	Run train.py


Original code: https://github.com/cesc-park/attend2u