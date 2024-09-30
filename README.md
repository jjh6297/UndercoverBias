# Undercover Bias

Implementation of "Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-wise Hidden Bias," ECCV 2024.


## Abstract

Public datasets play a crucial role in advancing data-centric AI, yet they remain vulnerable to illicit uses. This paper presents `undercover bias,' a novel dataset watermarking method that can reliably identify and verify unauthorized data usage. Our approach is inspired by an observation that trained models often inadvertently learn biased knowledge and can function on bias-only data, even without any information directly related to a target task. Leveraging this, we deliberately embed class-wise hidden bias via unnoticeable watermarks, which are unrelated to the target dataset but share the same labels. Consequently, a model trained on this watermarked data covertly learns to classify these watermarks. The model's performance in classifying the watermarks serves as irrefutable evidence of unauthorized usage, which cannot be achieved by chance. Our approach presents multiple benefits: 1) stealthy and model-agnostic watermarks; 2) minimal impact on the target task; 3) irrefutable evidence of misuse; and 4) improved applicability in practical scenarios. We validate these benefits through extensive experiments and extend our method to fine-grained classification and image segmentation tasks.


## Dependency


<!-- dependencies: -->


tensorflow >= 2.0




## Usage

1. you can train watermarking network by:
```
python TrainDWN.py
```


2. you can extract watermarked dataset by:
```
python get_Watermarked.py
python get_Watermarked_testData.py
```


3. you can evaluate Verification Ability on ResNet18:
```
python Evaluate_ResNet18.py
```
#


## Poster

<p align="center">
  <img src="https://github.com/jjh6297/UndercoverBias/blob/main/ECCV2024-poster_Watermarking.png"/>
</p>


## Citation

```
@inproceedings{jang2024Rethinking,
  title={Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-wise Hidden Bias},
  author={Jang, Jinhyeok and Han, ByungOk and Kim, Jaehong and and Youn, Chan-Hyun},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
