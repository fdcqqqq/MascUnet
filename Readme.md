

# **Multi-attention Brain Tumor Image Segmentation Algorithm Based on Unet**



## Installation

* Unbuntu 18.04
* Install TensorFlow 1.14 ,keras 2.1.5 and CUDA 9.0
* Clone this repo

```shell
git clone https://github.com/fdcqqqq/MascUnet
cd MascUnet
```

## Data Preparation

1. Prepare your dataset under the directory ```dataset``` .

  * Directory structure on new dataset needed for training and testing:
    * dataset/train_HGG
    * dataset/train_LGG
    * dataset/test_HGG
    * dataset/test_LGG

## Data preprocessing

* Modify paramter values in `data_processing.py`
* Run `./data_processing.py` to start data_processing.

```
python3 data_processing.py
```

## Train

* Modify paramter values in `./train.py`

* Run `./train.py` to start the train the model.

  ```shell
  python3 train.py
  ```

  

## Evaluate

* Specify the model path and test file path in `./predict.py`

* Run `./predict.py` to start the evaluation.

  ```shell
  python3 predict.py
  ```

* draw 3D nii predict images to 2D and show figures.

  ```shell
  python3 ./show_slicer_nii.py 
  ```

## results

1. Generate  images by following specifications under:

  * `./prediction`

## Figures

### Framework

<img src="figures\summary.png" style="zoom: 25%;" />

---

### Brats 2019 Dataset



<img src="figures\btats_data.png" style="zoom: 50%;" />



##### The Brats 2019 contains four modalities:**(a):** T2. **(b):** Flair. **(c):** T1.**(d):**T1c ,and label is **(e)**Grouth truth,the label include four type of tags,as shown in**(e)**,which are normal tissue(tag 0), necrosis and non-enhancing tumor(tag1),edema(tag 2),and enhancing tumor(tag 4).

---

### Parallel Dilation Convolution Feature Extraction Module

![](figures\unet.png)

---

### Masc Attention Module

<img src="figures\attention.png" style="zoom: 25%;" />

---

### Experiences

![](figures\compare.png)

---



<img src="figures\4.jpg" style="zoom: 50%;" />

---

## Note

* The repository is being updated
* Contact: Dian chenFu (17862003017@163.com)