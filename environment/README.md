# How to build environment

## Anaconda
**environment file: pytorch_wsi.yml**

1. Install Anaconda
2. conda env create -n new_env_name -f pytorch_wsi.yml

---
## Docker
future task...

---
## Other Info
### How to install openslide
https://github.com/openslide/openslide-python/issues/35
```
sudo apt-get install openslide-tools
sudo apt-get install python-openslide
pip install openslide-python
```


### How to use opencv
https://omohikane.com/python3_open_cv_libsm/
```
sudo apt install libsm6 libxrender1 libxext-dev
pip install opencv-python
```