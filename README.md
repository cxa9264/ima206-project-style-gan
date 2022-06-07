# ima206-project-style-gan
IMA206 project on style gan

Based on Pytorch implemented stylegan2-ada and InterFaceGan.

## StyleGan

Generate cifar-10 images
```
sh run_stylegan.sh cifar
```

Generate MetFaces images
```
sh run_stylegan.sh metfaces
```

Generate mixing example
```
sh run_stylegan.sh mixing
```

Images are generated at ```stylegan2-ada-pytorch/out/```

## InterFace Gan

Well.... it use the tf version StyleGan based on tf 1.10.0.....

1. Download StyleGan parameters and put in ```interfacegan/models/pretrain/```
```https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf```

2. 
NOT FINISH YET!
```
sh run_interfacegan 10000
```

### TODO
- [ ] upload generated images
- [ ] find a predictor to annotate generated images

### Inprogress
- [ ] generate images

### Done
...


