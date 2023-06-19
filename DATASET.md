# DATASET

Here we give instructions of downloading and preprocessing of the dataset. 

## Visual Genome

You can download the annotation directly by following steps. 

### Download:
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `/path/to/vg/VG_100K`. 

2. Download the [scene graphs annotations](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfI9vkdunDpCqp8ooxoHhloBE6KDuztZDWQM_Sbsw_1x5A?e=N8gWIS) and extract them to `/path/to/vg/vg_motif_anno`.

3. Link the image into the project folder
```
ln -s /path-to-vg datasets/vg
```

## Generate Training and Testing Data for each learning scenario 

The following steps need to be followed for the data generation for each scenario. 

1. 

