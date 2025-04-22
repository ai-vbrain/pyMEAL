# Multi-Encoder-Augmentation-Aware-Learning

Medical imaging is vital for diagnostics, yet clinical deployment remains hindered by patient variability, imaging artifacts, and poor model generalization. Deep learning has transformed image analysis, but its application to 3D imaging is limited by scarce high-quality data and inconsistencies from acquisition protocols, scanner variability, and motion artifacts. Traditional augmentation applies uniform transformations, overlooking the distinct characteristics of each type and struggling with large data volumes.

To overcome this, we introduce Multi-encoder Augmentation-Aware Learning (MEAL), a framework that processes four distinct augmentations through dedicated encoders. Applied to a CT-to-T1 MRI translation task, MEAL integrates three fusion strategies—concatenation (CC), fusion layer (FL), and adaptive controller block (BD)—to merge augmentation-specific features before decoding. MEAL-BD uniquely retains augmentation-aware representations during training, enabling protocol-invariant learning.

On non-augmented inputs, MEAL-CC improved SSIM and PSNR by 3.73% and 7.6%, respectively, outperforming standard augmentation by 2× and 4×. Under geometric transformations, MEAL-BD maintained high performance (SSIM: 0.81, PSNR: 24.52 dB), while baseline methods degraded sharply (SSIM < 0.5, PSNR < 18 dB). MEAL thus enhances model robustness and generalizability, advancing clinically reliable imaging solutions for tasks like surgical planning.

<img width="611" alt="Image" src="https://github.com/user-attachments/assets/2ce4b937-3a9d-4157-859f-10e379843efe" />
Fig. 1:Model architecture for the model having no augmentation and traditional augmentation


<img width="683" alt="Image" src="https://github.com/user-attachments/assets/811fc579-a0d0-4ebf-bd2b-e47b48405647" />

Fig. 2: Model architecture for Multi-Stream with a Build Controller method (BD), Fusion layer (FL) and Concatenation (CC)}
