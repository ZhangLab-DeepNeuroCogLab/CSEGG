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

python scripts/data_generation_s1.py --file_path "datasets/vg/"
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

python scripts/data_generation_s2.py --file_path "datasets/vg/"
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

python scripts/data_generation_s3.py --file_path "datasets/vg/"
```


