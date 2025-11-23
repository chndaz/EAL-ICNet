# EAL-ICNet
EAL-ICNet, a lightweight segmentation framework, integrates the advantages of both CNNs and Transformers to achieve a strong balance between segmentation accuracy, model compactness, and inference efficiency.
# Prepare data 

### Dataset descriptions 
To validate the effectiveness and efficiency of the proposed EAL-ICNet, we conducted comprehensive comparative experiments on circuit segmentation using enterprise-provided datasets, and further evaluated the model’s capability in via detection. In addition, two publicly available IC datasets were employed for broader performance assessment.

(1) OMC Dataset: The Optical Microscope Circuit (OMC) dataset consists of 3,084 optical microscope IC images with a resolution of $512 \times 512$, provided by an enterprise partner. Notably, high-quality and complete annotations were supplied by experienced engineers to facilitate the development of efficient and robust segmentation models. In our experiments, all images were divided into training, validation, and test sets with an 8:1:1 ratio.

(2) MIIC Dataset: The MIIC dataset is a large-scale collection specifically designed for automatic defect detection in IC images, containing a total of 25,276 high-resolution samples. Due to the vast data volume and partial redundancy among images, manual annotation of the entire dataset would incur significant costs. To ensure feasibility and representativeness, we randomly selected 3,446 images for manual pixel-level annotation and split them into training, validation, and test sets with an 8:1:1 ratio. The dataset is available at https://doi.org/10.21979/N9/WBLTFI.

(3) HY5SYN Dataset: The HY5SYN dataset consists of IC images fabricated with a 128 nm process and captured using an electron microscope. It is a dedicated dataset designed for precise segmentation of metal lines and via regions. The dataset contains 312 high-resolution labeled images with a resolution of $4096 \times 3536$, covering a variety of representative circuit structures and process layers. Due to computational resource limitations, we cropped all images into $1024 \times 1024$ patches, resulting in a total of 3,844 samples. These were also divided into training, validation, and test sets using an 8:1:1 ratio for model training and performance evaluation. The dataset is available at https://doi.org/10.17617/3.HY5SYN.

(4) OMV Dataset: To further assess the generalization ability of our model beyond circuit segmentation and its capability in recognizing via structures, we constructed the OMV(Optical Microscope Via) Dataset based on the original OMC dataset. Specifically, 473 images were randomly selected, and pixel-level annotations of via regions were manually provided by engineers. The dataset was split into training, validation, and test subsets in an 8:1:1 ratio to ensure experimental consistency and reliability.

<img width="4116" height="2083" alt="Fig1" src="https://github.com/user-attachments/assets/4cc9625e-eff9-4657-9027-22e28b6b9b81" />

### Experimental Setup and Implementation Detail
(1)Baselines: To comprehensively evaluate the effectiveness of the proposed EAL-ICNet, we compare it with a wide range of representative segmentation models, which can be categorized into two groups: lightweight networks and IC segmentation networks.

Lightweight networks: Several lightweight segmentation models are employed to assess the efficiency and compactness of our method, including IRDPNet, EDANet, ContextNet, ESNet, and LETNet. These models are characterized by reduced computational complexity and are widely used in real-time or resource-constrained scenarios. They serve as strong baselines for evaluating the trade-off between segmentation accuracy and inference speed.

IC segmentation networks: In addition, we involve multiple IC-oriented segmentation approaches for a more domain-specific comparison. These include FCN, UNet, HRNet, MAnet, and EUNet++, all of which have demonstrated strong capability in extracting fine structural details from IC or microscopic images. Among them, EUNet++ adopts a nested dense skip connection structure to enhance multi-scale feature fusion.
Finally, our proposed EAL-ICNet further integrates multi-scale feature aggregation and lightweight design principles to achieve a better balance between segmentation accuracy and computational efficiency on IC image datasets.

(2)Implementation Details: To ensure a fair comparison with existing studies, the input size of the HY5SYN dataset was uniformly set to $1024 \times 1024$, while the input sizes of the OMC, MIIC, and OMA datasets were adjusted to $512 \times 512$. It is worth noting that our TSSA module employs eight parallel attention heads. The model was optimized using the Adam optimizer, and all experiments were implemented with the PyTorch deep learning framework on an NVIDIA GeForce RTX 4060 Ti GPU with 16 GB of memory.
In our experiments, the segmentation results were directly generated as probability maps, and the final binary predictions were obtained by applying a threshold of 0.5. All experiments were randomly conducted three times, and the mean and standard deviation of each evaluation metric were recorded.


### Training Configuration

### Evaluation Metrics  
In the experiments, two evaluation metrics were used: Mean Intersection over Union (mIoU) and Mean Pixel Accuracy (MPA), to assess the quality of different networks. mIoU refers to the overlap rate between the generated candidate boxes and the original labeled boxes,~\emph{i.e.}, the intersection over union. A higher mIoU indicates better segmentation results.
Let $p_{ii}$ denote the number of correctly predicted elements, $p_{ij}$ the number of elements with true label $i$ and predicted label $j$, and $p_{ji}$ the number of elements with true label $j$ and predicted label $i$. Let $k$ represent the number of classes. Then, the Mean Intersection over Union (mIoU) can be expressed as:

$$
mIoU=\frac{1}{k+1}\sum_{i=0}^{k}\frac{p_{ii}}{\sum_{j=0}^{k}p_{ij}+\sum_{j=0}^{k}p_{ji}-p_{ii}}.
$$

Mean Pixel Accuracy (MPA) improves upon pixel accuracy by computing the pixel accuracy for each class individually and then averaging these accuracies across all classes. Let $k$ denote the number of classes, $p_{ii}$ be the total number of pixels with true class $i$ that are predicted as class $i$, and $p_{ij}$ be the total number of pixels with true class $i$ that are predicted as class $j$. MPA can be calculated using the following formula:

$$
MPA=\frac{1}{k+1}\sum_{i=0}^{k}\frac{p_{ii}}{\sum_{j=0}^{k}p_{ij}}.
$$


To further evaluate segmentation quality, we also employ the Dice coefficient (Dice), which measures the similarity between the predicted segmentation and the ground truth. Dice is particularly sensitive to class imbalance, making it suitable for IC segmentation where background pixels dominate. Let $TP$, $FP$, and $FN$ represent the number of true positive, false positive, and false negative pixels, respectively. The Dice coefficient is defined as:

$$
Dice = \frac{2TP}{2TP + FP + FN}.
$$

A higher Dice score indicates better overlap between prediction and ground truth.

In addition, the Aggregated Jaccard Index (AJI) is adopted as a connected-component-based evaluation metric to assess whether circuit segmentation results exhibit short-circuit or open-circuit issues. Unlike mIoU, which measures pixel-level overlap, AJI evaluates the Jaccard similarity across all connected regions, penalizing both over- and under-segmentation. A higher AJI indicates more accurate separation and connection of metal lines and vias, thus better preserving circuit topology.
Let $G = {G_1, G_2, \dots, G_n}$ and $S = {S_1, S_2, \dots, S_m}$ represent the sets of ground truth and segmented regions, respectively. For each $G_i$, let $S(G_i)$ denote the segmented region with the highest IoU. Then, the AJI is defined as:

$$
AJI = \frac{\sum_{i=1}^{n} |G_i \cap S(G_i)|}{\sum_{i=1}^{n} |G_i \cup S(G_i)| + \sum_{k \in U} |S_k|},
$$

where $U$ is the set of unmatched segmented instances. A higher AJI value indicates more accurate instance-level correspondence between prediction and ground truth.



