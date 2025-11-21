# EAL-ICNet
EAL-ICNet, a lightweight segmentation framework, integrates the advantages of both CNNs and Transformers to achieve a strong balance between segmentation accuracy, model compactness, and inference efficiency.EAL-ICNet是一种轻量级的分割框架，它融合了cnn和transformer的优点，在分割精度、模型紧凑性和推理效率之间取得了很好的平衡。

# Prepare data 

### Dataset descriptions 
To validate the effectiveness and efficiency of the proposed EAL-ICNet, we conducted comprehensive comparative experiments on circuit segmentation using enterprise-provided datasets, and further evaluated the model’s capability in via detection. In addition, two publicly available IC datasets were employed for broader performance assessment.

(1) OMC Dataset: The Optical Microscope Circuit (OMC) dataset consists of 3,084 optical microscope IC images with a resolution of $512 \times 512$, provided by an enterprise partner. Notably, high-quality and complete annotations were supplied by experienced engineers to facilitate the development of efficient and robust segmentation models. In our experiments, all images were divided into training, validation, and test sets with an 8:1:1 ratio.

(2) MIIC Dataset: The MIIC dataset is a large-scale collection specifically designed for automatic defect detection in IC images, containing a total of 25,276 high-resolution samples. Due to the vast data volume and partial redundancy among images, manual annotation of the entire dataset would incur significant costs. To ensure feasibility and representativeness, we randomly selected 3,446 images for manual pixel-level annotation and split them into training, validation, and test sets with an 8:1:1 ratio.

(3) HY5SYN Dataset: The HY5SYN dataset consists of IC images fabricated with a 128 nm process and captured using an electron microscope. It is a dedicated dataset designed for precise segmentation of metal lines and via regions. The dataset contains 312 high-resolution labeled images with a resolution of $4096 \times 3536$, covering a variety of representative circuit structures and process layers. Due to computational resource limitations, we cropped all images into $1024 \times 1024$ patches, resulting in a total of 3,844 samples. These were also divided into training, validation, and test sets using an 8:1:1 ratio for model training and performance evaluation.

(4) OMV Dataset: To further assess the generalization ability of our model beyond circuit segmentation and its capability in recognizing via structures, we constructed the OMV(Optical Microscope Via) Dataset based on the original OMC dataset. Specifically, 473 images were randomly selected, and pixel-level annotations of via regions were manually provided by engineers. The dataset was split into training, validation, and test subsets in an 8:1:1 ratio to ensure experimental consistency and reliability.

### Experimental Setup and Implementation Detail

### Training Configuration

### Evaluation Metrics  

## Model Architecture
