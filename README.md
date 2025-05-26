# pyMEAL: Multi-Encoder-Augmentation-Aware-Learning

pyMEAL is a multi-encoder framework for augmentation-aware learning that accurately performs CT-to-T1-weighted MRI translation under diverse augmentations. It utilizes four dedicated encoders and three fusion strategies, concatenation (CC), fusion layer (FL), and controller block (BD), to capture augmentation-specific features. MEAL-BD outperforms conventional augmentation methods, achieving SSIM > 0.83 and PSNR > 25 dB in CT-to-T1w translation.

<img width="611" alt="Image" src="https://github.com/user-attachments/assets/2ce4b937-3a9d-4157-859f-10e379843efe" />


Fig. 1:Model architecture for the model having no augmentation and traditional augmentation


<img width="683" alt="Image" src="https://github.com/user-attachments/assets/811fc579-a0d0-4ebf-bd2b-e47b48405647" />


Fig. 2: Model architecture for Multi-Stream with a Build Controller method (BD), Fusion layer (FL) and Concatenation (CC)

## Dependecies
tensorflow

matplotlib

SimpleITK

scipy

antspyx
