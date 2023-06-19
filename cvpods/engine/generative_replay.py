#write a class which takes in a list of words, based on those words loads the corresponding images, and then infer on those images and generate the corresponding ground truth 

import os
import sys
import torch
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

import shutil
from cvpods.engine import default_argument_parser, default_setup, launch

from cvpods.utils import comm



class GenerativeReplay(object):
    def __init__(self, list_of_predicates, path_of_images, path_of_model = None, path_of_output = None):

        #list_of_predicates = [ "_on", "for", "looking_at", "watching", "hanging_from", "painted_on", "playing", "made_of", "says", "flying_in" ]
        
        self.list_of_predicates = list_of_predicates
        self.path_of_images = path_of_images
        self.path_of_model = path_of_model
        self.path_of_output = path_of_output

        # self.cfg = get_cfg()
        # self.cfg.merge_from_file(self.path_of_model)
        # self.cfg.MODEL.WEIGHTS = self.path_of_model
        # self.cfg.freeze()

        # self.predictor = DefaultPredictor(self.cfg)


    def load_images(self):

        #load the images from all the folders which have any element of the list of predicates in their name

        list_of_folders = os.listdir(self.path_of_images)

        filtered_folders_final = []

        for i in range(len(self.list_of_predicates)):
            print(self.list_of_predicates[i])
            filtered_folders = []
            filtered_folders = [folder for folder in list_of_folders if any(pred in folder for pred in self.list_of_predicates) 
                    and len(folder.split(self.list_of_predicates[i])[0].split("_")) == 1 
                    and len(folder.split(self.list_of_predicates[i])[1].split("_")) == 1]
            
            # print(len(filtered_folders))
        
            filtered_folders_final.append(filtered_folders)
        
        filtered_folders_final = [item for sublist in filtered_folders_final for item in sublist]

        list_of_images = []

        for folder in filtered_folders_final:
            list_of_images.append(os.listdir(os.path.join(self.path_of_images, folder)))
        
        final_list_of_images = []

        for i in range(len(filtered_folders_final)):
            final_path = ""

            for j in range(len(list_of_images[i])):

                final_path = self.path_of_images + "/" + filtered_folders_final[i] + "/" + list_of_images[i][j]
            
            final_list_of_images.append(final_path)
        
        path_of_output = "/home/naitik2/projects/SGG_Continual/models/experiments/c2/cvpods/engine/gen_replay"

        for image in tqdm(final_list_of_images[10:20]):
            shutil.copy(os.path.join(self.path_of_images, image), path_of_output)

        print(len(final_list_of_images))

        return final_list_of_images
    
    def get_annotations(self):

        #get the annotations for the image

        # predictions = self.predictor(image_path)

        # print(predictions)

        # return predictions

        pass
    





#write the main function of the file to run the script

def main(args):

    list_of_predicates = [ "_near_", "_behind_", "_with_", "_holds_", "_above_", "_parked_on_", "_laying_on_", "_belonging_to_", "_eating_", "_and_"  ]
    path_of_images = "/home/naitik2/projects/image_gen_module/stable_diffusion_2_1/images"

    gen_replay = GenerativeReplay(list_of_predicates, path_of_images)

    gen_replay.load_images()



    # list_of_folders = os.listdir(path_of_images)

    # filtered_folders_final = []

    # for i in range(len(list_of_predicates)):
    #     print(list_of_predicates[i])
    #     filtered_folders = []
    #     filtered_folders = [folder for folder in list_of_folders if any(pred in folder for pred in list_of_predicates) 
    #                and len(folder.split(list_of_predicates[i])[0].split("_")) == 1 
    #                and len(folder.split(list_of_predicates[i])[1].split("_")) == 1]
        
    #     print(len(filtered_folders))
        
    #     filtered_folders_final.append(filtered_folders)

    # #make the nested list flat

    # filtered_folders_final = [item for sublist in filtered_folders_final for item in sublist]

    # #print 10 random elements from the list

    # print(random.sample(filtered_folders_final, 10))

    # print(filtered_folders_final)


    # list_of_folders = [x for x in list_of_folders if any(y in x for y in list_of_predicates)]

    # list_of_images = []

    # print(list_of_images[78:98])

    # for folder in list_of_folders:
    #     list_of_images.append(os.listdir(os.path.join(path_of_images, folder)))

    # print(list_of_images[78:98])

    # final_list_of_images = []

    # for i in range(len(list_of_folders)):
    #     final_path = ""

    #     for j in range(len(list_of_images[i])):

    #         final_path = "/home/naitik2/projects/image_gen_module/stable_diffusion_2_1/images/" + list_of_folders[i] + "/" + list_of_images[i][j]
        
    #     final_list_of_images.append(final_path)
    



    # # copy the images to the output folder

    # path_of_output = "/home/naitik2/projects/SGG_Continual/models/experiments/c2/cvpods/engine/gen_replay"

    # for image in tqdm(final_list_of_images[10:20]):
    #     shutil.copy(os.path.join(path_of_images, image), path_of_output)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(main, args.num_gpus, args=(args,))


