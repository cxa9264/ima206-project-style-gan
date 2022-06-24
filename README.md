# ima206-project-style-gan
IMA206 project on style gan

Based on Pytorch implemented InterFaceGan.

## InterFace Gan

Well.... it use the tf version StyleGan based on tf 1.10.0.....

* Manipulate attribute:
```
sh run_move.sh {latent_space_type} {attribute index}
```
Data is by default generated in ```./result/``` and the score is saved at ```./move_result```.

* Pretrained boundaries

Available at https://drive.google.com/file/d/1VENvrpsggwp0KLUUVHFObCTTXxBMMk1U/view?usp=sharing.
Put in ```./interfacegan/boundary```


* Train the SVM
```
sh run_interfacegan.sh {latent_space_type} {attribute index} {atribute name}
```

Example
```
sh run_interfacegan.sh w 20 Male
```
```
sh run_interfacegan.sh w 14 Double_Chin
```

## CNN Classifier
**References:**
- https://github.com/Hawaii0821/FaceAttr-Analysis

Prediction of face attributes:
```
sh run_predictor.sh {target_index}
```

- https://github.com/rgkannan676/Recognition-and-Classification-of-Facial-Attributes
- https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/discussion/78775





