# Spectral_Transformer





>[Learning to Segment Complex Vessel-like Structures with Spectral Transformer,


## usage
### datasets
Download the datasets and the files follows the following structure.


```
new_dataset
|_test
|   |_images
|   |	|_CFD
|   |   |  |_CFD_00..jpg
|   |   |_cracktree
|   |   |  |_a..jpg
|   |_masks
|   |	|_CFD
|   |   |  |_CFD_00..jpg
|   |   |_cracktree
|   |   |  |_a..jpg
|_train
|   |_images
|   |	|_CFD
|   |   |  |_CFD_00..jpg
|   |   |_cracktree
|   |   |  |_a..jpg
|   |_masks
|   |	|_CFD
|   |   |  |_CFD_00..jpg
|   |   |_cracktree
|   |   |  |_a..jpg
|   |_CFD.txt
|   |_cracktree.txt


        ......
```

train.txt format
```
    train_img_dir = "./datasets/new_dataset/train/"+ datasetName +".txt"   
    valid_img_dir = "./datasets/new_dataset/test/images/"+datasetName+'/'
    valid_lab_dir = "./datasets/new_dataset/test/masks/"+datasetName+'/'
.....
```

### Valid

Change the 'pretrain_dir','datasetName' and 'netName' in test.py

```
python test.py
```

### Pre-trained model

Download the trained model.

```
|-- model
    |-- <netname>
        |   |-- <trained_model.pkl>
        ......
```



| dataset    | Pre-trained Model                                            |
| ---------- | ------------------------------------------------------------ |
| CHASE-DB   | [Link] |
| PRIME-FP20 | [Link] |
| RECOVERY   | [Link] |
| cracktree  | [Link] |
| crackls315 | [Link] |
| CFD        | [Link] |
| Massachusetts Road   | [Link] |
| DeepGlobe Road       | [Link] |




## Acknowledgments

- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.


