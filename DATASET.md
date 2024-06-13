# DATASET
We adopt the same protocol with BGNN for Visual Genome and Openimage datasets.

## Visual Genome
The following is adapted from BGNN by following the same protocal of [Unbiased Scene Graph Generation from Biased Training](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) 
You can download the annotation directly by following steps.

### Download:
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `/path/to/vg/VG_100K`. 

2. Download the [scene graphs annotations](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfI9vkdunDpCqp8ooxoHhloBE6KDuztZDWQM_Sbsw_1x5A?e=N8gWIS) and extract them to `/path/to/vg/vg_motif_anno`.

3. Link the image into the project folder
```
ln -s /path-to-vg datasets/vg
```



## Openimage V4/V6 
We adopt Openimage datasets from BGNN.
### Download
1. The initial dataset(oidv6/v4-train/test/validation-annotations-vrd.csv) can be downloaded from [offical website]( https://storage.googleapis.com/openimages/web/download.html).

2. The Openimage is a very large dataset, however, most of images doesn't have relationship annotations. 
To this end, we filter those non-relationship annotations and obtain the subset of dataset ([.ipynb for processing](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EebESIOrpR5NrOYgQXU5PREBPR9EAxcVmgzsTDiWA1BQ8w?e=46iDwn) ). 

3. You can download the processed dataset: [Openimage V6(38GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EXdZWvR_vrpNmQVvubG7vhABbdmeKKzX6PJFlIdrCS80vw?e=uQREX3),
[Openimage V4(28GB)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EVWy0xJRx8RNo-zHF5bdANMBTYt6NvAaA59U32o426bRqw?e=6ygqFR)
3. By unzip the downloaded datasets, the dataset dir contains the `images` and `annotations` folder. 
Link the `open_imagev4` and `open_image_v6` dir to the `/datasets/openimages` then you are ready to go.
```
mkdir datasets/openimages
ln -s /path/to/open_imagev6 datasets/openimages
```

# DATASET

Here we give instructions of downloading and preprocessing of the dataset. 

## Visual Genome

You can download the images and notations directly by following steps. 

### Download:
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `path_to/vg/VG_100k_images`. 

2. Download the [scene graphs annotations](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfI9vkdunDpCqp8ooxoHhloBE6KDuztZDWQM_Sbsw_1x5A?e=N8gWIS) and extract them to `path_to/vg/vg_motif_anno`.

   Here, `path_to` is where you extract the dataset. (Size of the Dataset ~ 27Gb, choose the directory which has enough space).

3. Link the data folder to project folder using the following command:
```bash

ln -s path_to/vg datasets/vg
```

## Generate Training and Testing Data for each learning scenario 

### Learning Scenario S1 Dataset

1. Change the directory to `datasets` using the following command:
```bash

cd datasets/

```
2. Create `s1_data` directory using the following command:
```bash
mkdir s1_data

```
3. Create Learning Scenario S1 dataset using the following command:
```bash

python data_generation_s1.py --file_path "vg/vg_motif_anno/"
```

### Learning Scenario S2 Dataset

1. Change the directory to `datasets` using the following command:
```bash

cd datasets/

```
2. Create `s2_data` directory using the following command:
```bash
mkdir s2_data

```
3. Create Learning Scenario S2 dataset using the following command:
```bash

python data_generation_s2.py --file_path "vg/vg_motif_anno/"
```

### Learning Scenario S3 Dataset

1. Change the directory to `datasets` using the following command:
```bash

cd datasets/

```
2. Create `s3_data` directory using the following command:
```bash
mkdir s3_data

```
3. Create Learning Scenario S3 dataset using the following command:
```bash

python data_generation_s3.py --file_path "vg/vg_motif_anno/"
```


# Customized Dataset

Register you implement Dataset and Evaluator by editing the `cvpods/data/datasets/paths_route.py`.
```python
_PREDEFINED_SPLITS_VG_STANFORD_SGDET = {
    "dataset_type": "VGStanfordDataset",  # visual genome stanford split
    "evaluator_type": {
        "vgs": "vg_sgg",
    },
    "vgs": {
      # the former is image directry path, the later is annotation directry path
        "vgs_train": ("vg/VG_100k_images", "vg/vg_motif_anno"),
        "vgs_val": ("vg/VG_100k_images", "vg/vg_motif_anno"),
        "vgs_test": ("vg/VG_100k_images", "vg/vg_motif_anno"),

    }
}
```

### Now look at HOW_T0_USE.md for knowing various commands to run the train and eval scripts (especially if you are using multiple gpus)

More details refer to [cvpods tutorial](
https://github.com/Megvii-BaseDetection/cvpods/blob/master/docs/tutorials/cvpods%20tutorials.ipynb).
