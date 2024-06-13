#write a code to generate exemplars for RAS

import os
import sys
import torch
import numpy as np
import json
import torch
import spacy

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to("cuda")

nlp = spacy.load("en_core_web_lg")

with open('VG-SGG-dicts-with-attri.json') as f:
    data = json.load(f)
    idx_to_label = data['idx_to_label']
    idx_to_predicate = data['idx_to_predicate']
    object_count = data['object_count']
    predicate_count = data['predicate_count']
    attribute_count = data['attribute_count']
    print("Data loaded successfully")

#TODO : Write the code for Context Checker 

#TODO : Write the code for LTD Balancing 

#TODO : Write the code for annotation generation

#TODO : Write the code for saving the exemplars

#TODO : Write the code for selecting the triplets
    
def get_task_distributions():
    

    task_distribution_s1_rel = {1: [31, 16, 25, 47, 19, 34, 37, 27, 39, 15],
                            2: [20, 11, 7, 46, 33, 28, 2, 3, 17, 18],
                            3: [48, 38, 23, 41, 6, 36, 26, 32, 42, 45],
                            4: [30, 22, 40, 49, 43, 44, 13, 10, 4, 12],
                            5: [29, 8, 50, 21, 1, 24, 14, 5, 9, 35]} 


    task_distribution_s1_rel[1] = task_distribution_s1_rel[1]
    task_distribution_s1_rel[2] = task_distribution_s1_rel[2] + task_distribution_s1_rel[1]
    task_distribution_s1_rel[3] = task_distribution_s1_rel[3] + task_distribution_s1_rel[2]
    task_distribution_s1_rel[4] = task_distribution_s1_rel[4] + task_distribution_s1_rel[3]
    task_distribution_s1_rel[5] = task_distribution_s1_rel[5] + task_distribution_s1_rel[4]


    task_distribution_s2_obj = {1: [62, 17, 14, 100, 21, 49, 84, 143, 110, 43, 10, 98, 67, 93, 19, 64, 138, 18, 23, 95, 65, 52, 81, 11, 30, 41, 88, 37, 13, 32, 137, 77, 5, 83, 80, 33, 24, 63, 147, 136, 108, 12, 34, 15, 31, 55, 25, 109, 125, 71, 148, 142, 117, 29, 27, 69, 7, 94, 82, 2, 118, 8, 116, 70, 9, 85, 35, 47, 132, 146, 122, 72, 46, 103, 36, 133, 123, 140, 56, 101, 16, 131, 42, 6, 128, 92, 1, 39, 102, 89, 119, 106, 86, 107, 50, 51, 141, 68, 120, 79],
                            2: [90, 76, 73, 38, 87, 112, 40, 45, 28, 97, 124, 54, 66, 3, 114, 121, 144, 127, 44, 60, 48, 20, 59, 104, 113]
                            }    

    task_distribution_s2_rel = {1: [43, 40, 49, 41, 23, 7, 6, 19, 33, 16, 38, 11, 14, 46, 37, 13, 24, 4, 47, 5, 10, 9, 34, 3, 25, 17, 35, 42, 27, 12, 28, 39, 36, 2, 15, 44, 32, 26, 18, 45],
                                2: [31, 20, 22, 30, 48]}

    task_distribution_s2_obj[1] = task_distribution_s2_obj[1]
    task_distribution_s2_obj[2] = task_distribution_s2_obj[2] + task_distribution_s2_obj[1]

    task_distribution_s1_obj = task_distribution_s2_obj[2].copy()

    task_distribution_s2_rel[1] = task_distribution_s2_rel[1]
    task_distribution_s2_rel[2] = task_distribution_s2_rel[2] + task_distribution_s2_rel[1]

    task_distribution_s3_obj = {1: [123, 97, 36, 31, 122, 21, 91, 35, 132, 9, 19, 62, 79, 34, 128, 113, 61, 50, 139, 117, 111, 129, 73, 125, 148, 92, 109, 68, 47, 59],
                            2: [3, 110, 98, 76, 4, 78, 51, 102, 33, 115, 141, 126, 85, 133, 18, 37, 7, 149, 55, 143, 77, 25, 104, 96, 105, 24, 93, 58, 6, 53],
                            3: [45, 103, 81, 119, 86, 90, 65, 146, 121, 100, 32, 30, 39, 87, 134, 42, 41, 38, 66, 44, 89, 13, 71, 127, 75, 17, 88, 106, 142, 135],
                            4:[112, 116, 131, 95, 101, 69, 40, 99, 22, 60, 120, 136, 52, 8, 70, 20, 14, 12, 72, 1, 29, 16, 10, 118, 84, 144, 107, 15, 26, 56]
                            } 

    task_distribution_s3_obj[1] = task_distribution_s3_obj[1]
    task_distribution_s3_obj[2] = task_distribution_s3_obj[2] + task_distribution_s3_obj[1]
    task_distribution_s3_obj[3] = task_distribution_s3_obj[3] + task_distribution_s3_obj[2]
    task_distribution_s3_obj[4] = task_distribution_s3_obj[4] + task_distribution_s3_obj[3]   

    relation_classes_allowed_according_to_s3 = list(np.array([1,5,6,7,8,9,10,11,13,16,19,20,21,22,23,25,27,28,29,30,31,32,33,36,37,38,40,41,42,43,44,47,48,49,50]) )       

        
    def get_class_name_obj(obj_idx):
        return idx_to_label[str(obj_idx)]

    def get_class_name_rel(rel_idx):
        return idx_to_predicate[str(rel_idx)]

    # S1 task distribution

    for i in range(len(task_distribution_s1_rel)):
        for j in range(len(task_distribution_s1_rel[i+1])):
            task_distribution_s1_rel[i+1][j] = get_class_name_rel(task_distribution_s1_rel[i+1][j])

    for i in range(len(task_distribution_s1_obj)):
        task_distribution_s1_obj[i] = get_class_name_obj(task_distribution_s1_obj[i])

    # S2 task distribution
        
    for i in range(len(task_distribution_s2_rel)):
        for j in range(len(task_distribution_s2_rel[i+1])):
            task_distribution_s2_rel[i+1][j] = get_class_name_rel(task_distribution_s2_rel[i+1][j])

    for i in range(len(task_distribution_s2_obj)):
        for j in range(len(task_distribution_s2_obj[i+1])):
            task_distribution_s2_obj[i+1][j] = get_class_name_obj(task_distribution_s2_obj[i+1][j])

    # S3 task distribution
            
    for i in range(len(task_distribution_s3_obj)):
        for j in range(len(task_distribution_s3_obj[i+1])):
            task_distribution_s3_obj[i+1][j] = get_class_name_obj(task_distribution_s3_obj[i+1][j])

    for i in range(len(relation_classes_allowed_according_to_s3)):
        relation_classes_allowed_according_to_s3[i] = get_class_name_rel(relation_classes_allowed_according_to_s3[i])

    print("Task distribution loaded successfully")

    return [task_distribution_s1_rel, task_distribution_s1_obj, task_distribution_s2_rel, task_distribution_s2_obj, task_distribution_s3_obj, relation_classes_allowed_according_to_s3]


def triplet_selection():
    pass

def Context_Checker(decomposed_sgg):

    #for each triplet in decomposed_sgg, load the word embeddings of the objects and relations and check if the context is correct

    #if the context is correct, return True

    #else return False

    docs = []

    for i in range(len(decomposed_sgg)):

        obj_1 = decomposed_sgg[i][0]
        rel = decomposed_sgg[i][1]
        obj_2 = decomposed_sgg[i][2]

        #load the word embeddings of the objects and relationn from word2vec

        sentence = str(obj_1) + " " + str(rel) + " " + str(obj_2)
        docs.append(sentence)
    
    for i in range(len(docs)):
        for j in range(len(docs)):
            if i != j:
                doc1 = nlp(docs[i])
                doc2 = nlp(docs[j])
                if doc1.similarity(doc2) < 0.3:
                    return True

    return True

def gen_images_s1(task_distributions, required_images = 2000, num_images_per_sgg = 10):

    task_distribution_s1_rel = task_distributions[0]
    task_distribution_s1_obj = task_distributions[1]

    print("Generating images for S1")

    required_unique_sgg = required_images // num_images_per_sgg

    for i in range(len(task_distribution_s1_rel)):

        task_number = i + 1

        #generate a random number between 1 and 10

        scene_graph_complexity = np.random.randint(1, 10)

        decomposed_sgg = []

        while len(decomposed_sgg) < required_unique_sgg:

            semi_decomposed_sgg = []

            for j in range(scene_graph_complexity):

                #generate a random number between 1 and 5 with 4 included

                rel_class = np.random.randint(1, 5)

                #generate a random number between 1 and 50

                obj_class_1 = np.random.randint(1, 50)
                obj_class_2 = -1

                while True:
                    obj_class_2 = np.random.randint(1, 50)
                    if obj_class_2 != obj_class_1:
                        break
                
                if Context_Checker(semi_decomposed_sgg):
                
                    semi_decomposed_sgg.append([task_distribution_s1_obj[obj_class_1], task_distribution_s1_rel[i+1][rel_class], task_distribution_s1_obj[obj_class_2]])
            
            decomposed_sgg.append(semi_decomposed_sgg)

        prompt = []

        for i in range(len(decomposed_sgg)):
            for j in range(len(decomposed_sgg[i])):
                semi_prompt = "Realistic Image of"
                for k in range(len(decomposed_sgg[i][j])):
                    semi_prompt = " " + decomposed_sgg[i][j][k] + " "
                
                prompt.append(semi_prompt + "and")
        
        for i in range(num_images_per_sgg):
            
            image = pipe(prompt).images[0]
            image.save("s1/image_" + str(task_number) + str(i) + ".png")
        


def gen_images_s2(task_distributions, required_images = 1500, num_images_per_sgg = 10):

    task_distribution_s2_rel = task_distributions[2]
    task_distribution_s2_obj = task_distributions[3]

    print("Generating images for S2")

    required_unique_sgg = required_images // num_images_per_sgg

    for i in range(len(task_distribution_s2_rel)):

        task_number = i + 1

        #generate a random number between 1 and 10

        scene_graph_complexity = np.random.randint(1, 10)

        decomposed_sgg = []

        while len(decomposed_sgg) < required_unique_sgg:

            semi_decomposed_sgg = []

            for j in range(scene_graph_complexity):

                #generate a random number between 1 and 5 with 4 included

                rel_class = np.random.randint(1, 5)

                #generate a random number between 1 and 50

                obj_class_1 = np.random.randint(1, 50)
                obj_class_2 = -1

                while True:
                    obj_class_2 = np.random.randint(1, 50)
                    if obj_class_2 != obj_class_1:
                        break
                
                if Context_Checker(semi_decomposed_sgg):
                
                    semi_decomposed_sgg.append([task_distribution_s2_obj[obj_class_1], task_distribution_s2_rel[i+1][rel_class], task_distribution_s2_obj[obj_class_2]])
            
            decomposed_sgg.append(semi_decomposed_sgg)

        prompt = []

        for i in range(len(decomposed_sgg)):
            for j in range(len(decomposed_sgg[i])):
                semi_prompt = "Realistic Image of"
                for k in range(len(decomposed_sgg[i][j])):
                    semi_prompt = " " + decomposed_sgg[i][j][k] + " "
                
                prompt.append(semi_prompt + "and")
        
        for i in range(num_images_per_sgg):
            
            image = pipe(prompt).images[0]
            image.save("s2/image_" + str(task_number) + str(i) + ".png")

   

def gen_images_s3(task_distributions, required_images = 2000, num_images_per_sgg = 10):

    task_distribution_s3_obj = task_distributions[4]
    relation_classes_allowed_according_to_s3 = task_distributions[5]

    print("Generating images for S3")

    required_unique_sgg = required_images // num_images_per_sgg

    for i in range(len(relation_classes_allowed_according_to_s3)):

        task_number = i + 1

        #generate a random number between 1 and 10

        scene_graph_complexity = np.random.randint(1, 10)

        decomposed_sgg = []

        while len(decomposed_sgg) < required_unique_sgg:

            semi_decomposed_sgg = []

            for j in range(scene_graph_complexity):

                #generate a random number between 1 and 5 with 4 included

                rel_class = np.random.randint(1, 5)

                #generate a random number between 1 and 50

                obj_class_1 = np.random.randint(1, 50)
                obj_class_2 = -1

                while True:
                    obj_class_2 = np.random.randint(1, 50)
                    if obj_class_2 != obj_class_1:
                        break
                
                if Context_Checker(semi_decomposed_sgg):
                
                    semi_decomposed_sgg.append([task_distribution_s3_obj[obj_class_1], relation_classes_allowed_according_to_s3[i+1][rel_class], task_distribution_s3_obj[obj_class_2]])
            
            decomposed_sgg.append(semi_decomposed_sgg)

        prompt = []

        for i in range(len(decomposed_sgg)):
            for j in range(len(decomposed_sgg[i])):
                semi_prompt = "Realistic Image of"
                for k in range(len(decomposed_sgg[i][j])):
                    semi_prompt = " " + decomposed_sgg[i][j][k] + " "
                
                prompt.append(semi_prompt + "and")
        
        for i in range(num_images_per_sgg):
            
            image = pipe(prompt).images[0]
            image.save("s3/image_" + str(task_number) + str(i) + ".png")

    


def gen_images(scenario, task_distributions):

    if scenario == "s1" :
        gen_images_s1(task_distributions)
    
    if scenario == "s2" :
        gen_images_s2(task_distributions)
    
    if scenario == "s3" :
        gen_images_s3(task_distributions)



    
#write the main function to run the python file 

if __name__ == "__main__":

    task_distributions = get_task_distributions()

    gen_images("s1", task_distributions)

    gen_images("s2", task_distributions)

    gen_images("s3", task_distributions)
    
    print("Images generated successfully")
  





