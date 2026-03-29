**Author:** Junyi Shen

**Date:** April 13, 2025  

## Overview
Implemented three models on the Oxford Flowers dataset:  

1. **Convolutional Neural Network (CNN)** – classifies images.  
2. **Auto-encoder** – encodes images into latent space and reconstructs them.  
3. **UNet Latent De-noising Model** – generates clear flower images from random noise over multiple de-noising steps.  

---

## Task 1: Convolutional Neural Network

**Dataset:** Oxford Flowers (coarse: 10 labels, fine: 102 labels).  
| Dataset | Train | Validation | Test |
|---------|-------|------------|------|
| Coarse  | 2322  | 390        | 390  |
| Fine    | 6149  | 1020       | 1020 |

**Image size:** 96×96  

**CNN Architecture:**  
| Layer Type         | Filters/Neurons | Kernel | Stride | Padding | Output Size         |
|-------------------|----------------|-------|--------|--------|-------------------|
| Input             | –              | –     | –      | –      | 96×96×3           |
| Conv1             | 16             | 3×3   | 1      | 1      | 96×96×16          |
| Max Pool1         | –              | 2×2   | 2      | –      | 48×48×16          |
| Conv2             | 32             | 3×3   | 1      | 1      | 48×48×32          |
| Max Pool2         | –              | 2×2   | 2      | –      | 24×24×32          |
| Conv3             | 64             | 3×3   | 1      | 1      | 24×24×64          |
| Max Pool3         | –              | 2×2   | 2      | –      | 12×12×64          |
| Conv4             | 128            | 3×3   | 1      | 1      | 12×12×128         |
| Max Pool4         | –              | 2×2   | 2      | –      | 6×6×128           |
| Conv5             | 256            | 3×3   | 1      | 1      | 6×6×256           |
| Max Pool5         | –              | 2×2   | 2      | –      | 3×3×256           |
| FC1               | 128            | –     | –      | –      | 128               |
| FC2               | 512            | –     | –      | –      | 512               |
| Output            | 10/102         | –     | –      | –      | 10 (coarse) / 102 (fine) |

**Results:**  
- Coarse: 81% test accuracy  
- Fine: 72% test accuracy  
<img width="1482" height="1352" alt="image" src="https://github.com/user-attachments/assets/dc498abf-53c8-49fd-8e61-4093a5f2aa65" />
<img width="1470" height="1324" alt="image" src="https://github.com/user-attachments/assets/c50c8326-6bdb-4dd3-8556-acd8e5da3dc5" />


---

## Task 2a: Auto-encoder

**Auto-encoder Architecture:**  

**Encoder:**  
| Layer Type | In Channels | Out Channels | Kernel | Stride | Padding | Output Size  |
|-----------|-------------|--------------|-------|--------|--------|--------------|
| Input     | –           | –            | –     | –      | –      | 96×96×3      |
| Conv1     | 3           | 16           | 3×3   | 2      | 1      | 48×48×16     |
| Conv2     | 16          | 32           | 3×3   | 2      | 1      | 24×24×32     |
| Conv3     | 32          | 64           | 3×3   | 2      | 1      | 12×12×64     |

**Decoder:**  
| Layer Type           | In Channels | Out Channels | Kernel | Stride | Padding | Output Size |
|---------------------|-------------|--------------|-------|--------|--------|-------------|
| Transpose Conv1     | 64          | 32           | 3×3   | 2      | 1      | 24×24×32    |
| Transpose Conv2     | 32          | 16           | 3×3   | 2      | 1      | 48×48×16    |
| Transpose Conv3     | 16          | 3            | 3×3   | 2      | 1      | 96×96×3     |

**Training:**  
- Epochs: 150  
- Loss: MSE  
- Batch size: 16  
- Learning rate: 0.0001

**Results:**  
<img width="1466" height="336" alt="image" src="https://github.com/user-attachments/assets/26e2f118-58f2-42ac-b1e9-46de83306968" />


---

## Task 2b: UNet De-noising Model

**UNet Architecture:**  
| Layer Name | Layer Type          | In Channels | Out Channels | Kernel | Stride | Output Size |
|-----------|-------------------|------------|--------------|-------|--------|-------------|
| Input     | –                 | –          | –            | –     | –      | 12×12×64   |
| Down1_1   | Conv               | 64         | 128          | 3×3   | 1      | 12×12×128  |
| Down1_2   | Conv               | 128        | 128          | 3×3   | 1      | 12×12×128  |
| Down1_3   | Conv               | 128        | 128          | 3×3   | 1      | 12×12×128  |
| Pool1     | MaxPool            | –          | –            | 2×2   | 2      | 6×6×128    |
| Down2_1   | Conv               | 128        | 256          | 3×3   | 1      | 6×6×256    |
| Down2_2   | Conv               | 256        | 256          | 3×3   | 1      | 6×6×256    |
| Down2_3   | Conv               | 256        | 256          | 3×3   | 1      | 6×6×256    |
| Pool2     | MaxPool            | –          | –            | 2×2   | 2      | 3×3×256    |
| Bottleneck1 | Conv             | 256        | 512          | 3×3   | 1      | 3×3×512    |
| Bottleneck2 | Conv             | 512        | 1024         | 3×3   | 1      | 3×3×1024   |
| Bottleneck3 | Conv             | 1024       | 512          | 3×3   | 1      | 3×3×512    |
| Bottleneck4 | Conv             | 512        | 256          | 3×3   | 1      | 3×3×256    |
| Up2_1     | Transpose Conv     | 256        | 256          | 3×3   | 2      | 6×6×256    |
| Up2_2     | Conv               | 256        | 256          | 3×3   | 1      | 6×6×256    |
| Up2_3     | Conv               | 256+256    | 128          | 3×3   | 1      | 6×6×128    |
| Up1_1     | Transpose Conv     | 128        | 128          | 3×3   | 2      | 12×12×128  |
| Up1_2     | Conv               | 128        | 128          | 3×3   | 1      | 12×12×128  |
| Up1_3     | Conv               | 128+128    | 64           | 3×3   | 1      | 12×12×64   |
| Output    | Conv               | 64         | 64           | 3×3   | 1      | 12×12×64   |

**Training:**  
- Learning rate: 0.0001  
- Batch size: 16  
- Epochs: 1000  
- Input noise levels: 10
<img width="1506" height="246" alt="image" src="https://github.com/user-attachments/assets/18d3ffff-9408-4f76-929d-55b87f20cacb" />


**Results:**  
- Generates 96×96 images from random noise over 10 de-noising steps  
- Handles diverse flower types with correct color distributions  
<img width="1498" height="1366" alt="image" src="https://github.com/user-attachments/assets/b6864545-fb45-4573-8f28-f9bf77b12272" />


---

## Conclusion
- Developed CNN for classification, auto-encoder for reconstruction, UNet for image generation.  
- Key lessons: hyper-parameter tuning, regularization, and iterative debugging lead to better results.
