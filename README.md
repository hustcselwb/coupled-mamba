# Coupled Mamba: Enhanced Multi-modal Fusion with Coupled State Space Model（NIPS 2024)
<div align="center">
  Wenbing Li, Hang Zhou, Junqing Yu, Zikai Song, Wei Yang

  I'm very pleased that our work has been accepted by NeurIPS 2024. 
</div>

## Abstract
![pipline](https://github.com/hustcselwb/coupled-mamba/blob/main/pipline.png)
The core of multi-modal fusion is to leverage the complementary information from different modalities. Existing methods often rely on traditional neural architectures, which struggle to capture complex interactions between modalities. Recent advances in State Space Models (SSMs), such as the Mamba model, have shown promise for stronger fusion. Despite this, SSMs face challenges in fusing multiple modalities due to hardware parallelism constraints.

To address this, we propose the Coupled SSM model, which links the state chains of multiple modalities while keeping intra-modality processes independent. Our model includes an inter-modal state transition mechanism, where the current state depends on both its own chain and neighboring chains' states from the previous time step. We also develop a global convolution kernel to support hardware parallelism.

Extensive experiments on datasets CMU-MOSEI, CH-SIMS, and CH-SIMSV2 show that our model improves F1-Score by 0.4%, 0.9%, and 2.3%, respectively, offers 49% faster inference, and saves 83.7% GPU memory compared to current state-of-the-art methods. These results demonstrate the effectiveness of the Coupled Mamba model in enhancing multi-modal fusion.

## Paper
We will promptly update with the latest paper link after submitting the camera-ready version of the [paper](https://arxiv.org/abs/2405.18014).

## Code Coming Soon
We will release the code soon.Including the model,dataset,eval code. 

## Dataset
[CMU-MOSEI](https://aclanthology.org/P18-1208.pdf), [CH_SIMS](https://aclanthology.org/2020.acl-main.343/), CH_SIMSV2, BRCA, MM-IMDB, datasets link are here 

## Environment
```
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```
## Citation
```
@article{li2024coupled,
  title={Coupled Mamba: Enhanced Multi-modal Fusion with Coupled State Space Model},
  author={Li, Wenbing and Zhou, Hang and Song, Zikai and Yang, Wei},
  journal={arXiv preprint arXiv:2405.18014},
  year={2024}
}
```
