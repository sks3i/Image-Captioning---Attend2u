Image Captioning - Attend2u <br />
---

Implementation of Attend to You: Personalized Image Captioning with Context Sequence Memory Networks <br />

* Work in progress *

### Requirements
* Python 3.5
* TensorFlow > 1.10
* Other requirements - check requirements.txt

### Installation
- `conda create -name attend2u python=3.5 pip`
- `conda activate attend2u`
- `pip install -r requirements.txt`

### Dataset
Download Instagram dataset. <br />
&nbsp;&nbsp;&nbsp;&nbsp;[JSON](https://drive.google.com/uc?export=download&id=0B3xszfcsfVUBdG0tU3BOQWV0a0E) file & 
[Images](https://drive.google.com/uc?export=download&id=0B3xszfcsfVUBVkZGU2oxYVl6aDA) <br />

Save the files to ${project_root}/data

Example of /data directory structure
```bash
├── data
│   ├── caption_dataset
│   ├── hashtag_dataset
│   ├── images
│   ├── json
│   ├── resnet_pool5_features

```
### Running Code

1. Download ResNet trained model <br/>
&nbsp;&nbsp;&nbsp;&nbsp;Run scripts/download_pretrained_resnet_101.sh

2. Extract images features <br/>
&nbsp;&nbsp;&nbsp;&nbsp;Run scripts/extract_features.sh
	
3. Configure the network <br/>
&nbsp;&nbsp;&nbsp;&nbsp;Edit model parameters, training/evaluation parameters and data paths in configs.py
	
4. Train <br/>
&nbsp;&nbsp;&nbsp;&nbsp;Run train.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python train.py

5. Evaluate <br />
&nbsp;&nbsp;&nbsp;&nbsp;Run eval.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python eval.py

---

Original code : https://github.com/cesc-park/attend2u

Coco evaluation tools for Python 3 : https://github.com/Illuminati91/pycocoevalcap
