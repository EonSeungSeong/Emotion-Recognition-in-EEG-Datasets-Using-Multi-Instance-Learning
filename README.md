# Emotion Recognition in EEG Dataset(GAMEEMO) using Multi-Instance Learning
This repo is a term project from the “AI for Electroencephalography data” major class at Pusan National University. **It was done in a short period of time and may contain inaccurate information.**

## About
This repo is a project that applies the model from the paper “Attention-based Deep Multiple Instance Learning” to an EEG dataset.
This project is implemented in Python.

## Model Architecture
![2dcnn_arch](https://github.com/EonSeungSeong/Emotion-Recognition-in-EEG-Datasets-Using-Multi-Instance-Learning/blob/main/assets/assets/AttentionMIL2D.png)

![1dcnn_arch](https://github.com/EonSeungSeong/Emotion-Recognition-in-EEG-Datasets-Using-Multi-Instance-Learning/blob/main/assets/assets/AttentionMIL1D.png)

## Installation & Run
```bash
pip install -r requirements.txt
python baseline.py
```
> **Note**: wheels are available for Linux and MacOS. & need to have a Wandb account.

## Visualization
![G3](https://github.com/EonSeungSeong/Emotion-Recognition-in-EEG-Datasets-Using-Multi-Instance-Learning/blob/main/assets/assets/bag_17_class_2_attention_heatmap.png)
> G3 : Horror

![G4](https://github.com/EonSeungSeong/Emotion-Recognition-in-EEG-Datasets-Using-Multi-Instance-Learning/blob/main/assets/assets/bag_7_class_3_attention_heatmap.png)
> G4 : Funny

![G1](https://github.com/EonSeungSeong/Emotion-Recognition-in-EEG-Datasets-Using-Multi-Instance-Learning/blob/main/assets/assets/bag_6_class_0_attention_heatmap.png)
> G1 : Boring

![G2](https://github.com/EonSeungSeong/Emotion-Recognition-in-EEG-Datasets-Using-Multi-Instance-Learning/blob/main/assets/assets/bag_7_class_1_attention_heatmap.png)
> G2 : Calm

## Citation
```
@article{ITW:2018,
  title={Attention-based Deep Multiple Instance Learning},
  author={Ilse, Maximilian and Tomczak, Jakub M and Welling, Max},
  journal={arXiv preprint arXiv:1802.04712},
  year={2018}
}
```

```
@article{alakus2020database,
  title={Database for an emotion recognition system based on EEG signals and various computer games--GAMEEMO},
  author={Alakus, Talha Burak and Gonen, Murat and Turkoglu, Ibrahim},
  journal={Biomedical Signal Processing and Control},
  volume={60},
  pages={101951},
  year={2020},
  publisher={Elsevier}
}
```

## Related Links
* https://www.kaggle.com/datasets/sigfest/database-for-emotion-recognition-system-gameemo
* https://github.com/AMLab-Amsterdam/AttentionDeepMIL