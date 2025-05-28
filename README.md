# pyMEAL: Multi-Encoder-Augmentation-Aware-Learning

pyMEAL is a multi-encoder framework for augmentation-aware learning that accurately performs CT-to-T1-weighted MRI translation under diverse augmentations. It utilizes four dedicated encoders and three fusion strategies, concatenation (CC), fusion layer (FL), and controller block (BD), to capture augmentation-specific features. MEAL-BD outperforms conventional augmentation methods, achieving SSIM > 0.83 and PSNR > 25 dB in CT-to-T1w translation.

<img width="611" alt="Image" src="https://github.com/user-attachments/assets/2ce4b937-3a9d-4157-859f-10e379843efe" />


Fig. 1:Model architecture for the model having no augmentation and traditional augmentation


<img width="683" alt="Image" src="https://github.com/user-attachments/assets/811fc579-a0d0-4ebf-bd2b-e47b48405647" />


Fig. 2: Model architecture for Multi-Stream with a Builder Controller block method (BD), Fusion layer (FL) and Encoder concatenation (CC)

## Dependecies
tensorflow

matplotlib

SimpleITK

scipy

antspyx

## Download Model Files

You can download any `.h5` file directly:

- [Download builder1_mode1l1abW512_1_11211z1p1rt_.h5](https://huggingface.co/AI-vBRAIN/pyMEAL/resolve/main/builder1_mode1l1abW512_1_11211z1p1rt_.h5)
- [Download best_moderRl_RHID2_1mo.h5](https://huggingface.co/AI-vBRAIN/pyMEAL/resolve/main/best_moderRl_RHID2_1mo.h5)
- [Download bestac22_mode3l_512m2_m21.h5](https://huggingface.co/AI-vBRAIN/pyMEAL/resolve/main/bestac22_mode3l_512m2_m21.h5)
- [Download direct7_11ag23f11.h5](https://huggingface.co/AI-vBRAIN/pyMEAL/resolve/main/direct7_11ag23f11.h5)
- [Download best_modelaf2ndab7_221ag12g11.h5](https://huggingface.co/AI-vBRAIN/pyMEAL/resolve/main/best_modelaf2ndab7_221ag12g11.h5)


```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

my_folder = "./my_models"  # or any path you want

model_path = hf_hub_download(
    repo_id="AI-vBRAIN/pyMEAL",
    filename="builder1_mode1l1abW512_1_11211z1p1rt_.h5",  # or any other desired model in our Huggingface
    repo_type="model",
    cache_dir=my_folder
)

# Load the model from that path
model = tf.keras.models.load_model(model_path, compile=False)
# Run inference
output = model.predict(input_data)
```
Here, `input_data` refers to a CT image, and the corresponding T1-weighted (T1w) image is predicted as the output.

For detailed instructions on how to use each module of the **pyMEAL** software, please refer to the [tutorial section on our GitHub repository](https://github.com/ai-vbrain/pyMEAL).

## How to get support?
For help, contact:

Dr. Ilyas (<amoiIyas@hkcoche.org>) or Dr. Maradesa (<amaradesa@hkcoche.org>)


