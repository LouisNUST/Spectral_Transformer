# Spectral_Transformer





>[Learning to Segment Complex Vessel-like Structures with Spectral Transformer,


## usage
### datasets
Download the [cracktree](https://www.sciencedirect.com/science/article/pii/S0167865511003795),[crackls315](Deepcrack: Learning hierarchi-
cal convolutional features for crack detection.),[stone331](https://www.sciencedirect.com/science/article/pii/S1051200420302529),[crack537](https://www.sciencedirect.com/science/article/pii/S0925231219300566) dataset and the file follows the following structure.

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


                                                       |

## Acknowledgments

- We thank the anonymous reviewers for valuable and inspiring comments and suggestions.

## Reference
>``` 
>@InProceedings{Liu_2021_ICCV,
>author    = {Liu, Huajun and Miao, Xiangyu and Mertz, Christoph and Xu, Chengzhong and Kong, Hui},
>title     = {CrackFormer: Transformer Network for Fine-Grained Crack Detection},
>booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
>month     = {October},
>year      = {2021},
>pages     = {3783-3792}
>}
>
>@article{Liu2023CrackFormer,
>title={CrackFormer Network for Pavement Crack Segmentation},
>author={Huajun Liu and Jing Yang and Xiangyu Miao and Christoph Mertz and Hui Kong},
>journal={IEEE Transactions on Intelligent Transportation Systems},
>volume={24},
>number={9},
>pages={9240-9252},
>year={2023},
>publisher={IEEE}
>}
>
>```
