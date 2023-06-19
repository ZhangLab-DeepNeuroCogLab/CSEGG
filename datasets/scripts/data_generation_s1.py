import h5py
import numpy as np 
from tqdm import tqdm
import pickle
import random
import os
import shutil

from cvpods.engine import default_argument_parser


def filter_c2(gt_classes, boxes, relationships, attributes, class_allowed):
    
    gt_classes_final = [] 
    boxes_final = [] 
    relationships_final = [] 
    attributes_final = []
    
    for i in range(len(gt_classes)):
        gt_classes_final.append(gt_classes[i])
        boxes_final.append(boxes[i])
    
    for j in range(relationships.shape[0]):
        
        rel_tuple = relationships[j]
        rel_id = rel_tuple[2]

        if rel_id in class_allowed :
            relationships_final.append(rel_tuple)
    

    relationships_final = np.array(relationships_final)
    
    for k in range(attributes.shape[0]):
        attributes_final.append(attributes[k])
    
    attributes_final = np.array(attributes_final)
    
    
    return gt_classes_final, boxes_final, relationships_final, attributes_final

def divide_dataset_task(task_number, class_allowed, dataset_original):
    
    dataset_final = {"boxes" : [], "relationships"  : [], "gt_classes" : [], "gt_attributes" : [], "split_mask" : []}
    
    index = 0

    print(class_allowed)
    
    split_mask_final = []

    for i in range(len(dataset_original["split_mask"])):
        
        if dataset_original["split_mask"][i] :
            
            gt_classes_i = dataset_original["gt_classes"][index]
            boxes_i = dataset_original["boxes"][index]
            relationships_i = dataset_original["relationships"][index]
            attributes_i = dataset_original["gt_attributes"][index]
            
            
            gt_classes_final_i, boxes_final_i, relationships_final_i, attributes_final_i = filter_c2(gt_classes_i, 
                                                                                                     boxes_i, 
                                                                                                     relationships_i, 
                                                                                                     attributes_i,
                                                                                                     class_allowed)
            # print("Lenght of the relationships ",len(relationships_final_i))
            
            if len(relationships_final_i) > 0 :
                split_mask_final.append(True)
                dataset_final["boxes"].append(boxes_final_i)
                dataset_final["relationships"].append(relationships_final_i)
                dataset_final["gt_classes"].append(gt_classes_final_i)
                dataset_final["gt_attributes"].append(attributes_final_i)
            else :
                split_mask_final.append(False)
                
            index = index + 1  
        else :
            split_mask_final.append(False)
        
        dataset_final["split_mask"] = split_mask_final
    # print(index)
    return dataset_final

def divide_dataset(task_distribution, dataset_dicts):
    
    divided_dataset_dict = {}
    
    for task_number, class_allowed in task_distribution.items():
        divided_dataset_dict[task_number] = {}
    
    
    for task_number, _ in divided_dataset_dict.items():
        
        # print(divide_dataset_dict[task_number])
        
        divided_dataset_dict[task_number] = divide_dataset_task(task_number, task_distribution[task_number], dataset_dicts)
    
    return divided_dataset_dict

def flatten(list_of_lists):
    flat_list = [np.array(item) for sublist in list_of_lists for item in sublist]
    
    return flat_list

def generate_first_and_last_arr(dataset_dict):
    
    img_to_first_box = []
    img_to_last_box = []

    prev_start_box = -1
    prev_end_box = -1 
    current_start_box = 0
    current_end_box = 0

    index = 0

    for i in range(len(dataset_dict["split_mask"])):


        if dataset_dict["split_mask"][i] :

            gt_classes = dataset_dict["gt_classes"][index]

            current_start_box = prev_end_box + 1

            current_end_box = current_start_box + len(gt_classes) - 1

            img_to_first_box.append(current_start_box)
            img_to_last_box.append(current_end_box)

            prev_start_box = current_start_box 
            prev_end_box = current_end_box

            index = index + 1

        else :

            img_to_first_box.append(-1)
            img_to_last_box.append(-1)


    img_to_first_rel = []
    img_to_last_rel = []

    prev_start_rel = -1
    prev_end_rel = -1 
    current_start_rel = 0
    current_end_rel = 0

    relationships_present = []

    predicate = []

    index = 0
    
    i = 0 

    for i in range(len(dataset_dict["split_mask"])):

        if dataset_dict["split_mask"][i] :

            relations = dataset_dict["relationships"][index]

            # print(relations.shape[0])

            if relations.shape[0] == 0 :
                img_to_first_rel.append(-1)
                img_to_last_rel.append(-1)

                index = index + 1

            else : 

                current_start_rel = prev_end_rel + 1
                current_end_rel = current_start_rel + relations.shape[0] - 1

                img_to_first_rel.append(current_start_rel)
                img_to_last_rel.append(current_end_rel)


                prev_start_rel = current_start_rel
                prev_end_rel = current_end_rel

                index = index + 1

                relationships_present.append(relations[:,0:2] + img_to_first_box[i])
                predicate.append(relations[:,-1])

        else :
            img_to_first_rel.append(-1)
            img_to_last_rel.append(-1)
        
        
    return np.array(img_to_first_box), np.array(img_to_first_rel), np.array(img_to_last_box), np.array(img_to_last_rel), flatten(predicate), flatten(relationships_present)

def convert_to_h5(dataset_dict, task_number, split_str, test_combined, exemp) :
    
    img_to_first_box, img_to_first_rel, img_to_last_box, img_to_last_rel, predicate, relationships_present = generate_first_and_last_arr(dataset_dict)
    
    #flattening operation 
    attributes = np.array(flatten(dataset_dict["gt_attributes"]))
    boxes_1024 = np.array(flatten(dataset_dict["boxes"]))
    labels = np.array(flatten(dataset_dict["gt_classes"])) + 1 
    labels = labels.reshape(labels.shape[0],1)
    
    predicate = np.array(predicate)
    predicate = predicate.reshape(predicate.shape[0],1)
    
    
    relationships_present = np.array(relationships_present)
    
    split = []
    
    for i in range(len(dataset_dict["split_mask"])):
        if dataset_dict["split_mask"][i] :
            split.append(0)
        else :
            split.append(-1)
    
    split = np.array(split)
    
    if split_str == "test" :

        if test_combined :
            hf = h5py.File("s1_data/s1_task_" + str(task_number) + "/" + "s1_task_" + str(task_number) + "_" + split_str + "_cumm_" +".h5", 'w')
        else :
            hf = h5py.File("s1_data/s1_task_" + str(task_number) + "/" + "s1_task_" + str(task_number) + "_" + split_str +".h5", 'w')
    else :
        if exemp == 10 :
            hf = h5py.File("s1_data/s1_task_" + str(task_number) + "_exemp_10" + "/" + "s1_task" + str(task_number) + ".h5", 'w')
        elif exemp == 100 :
            hf = h5py.File("s1_data/s1_task_" + str(task_number) + "_exemp_100" + "/" + "s1_task" + str(task_number) + ".h5", 'w')
        else :
            hf = h5py.File("s1_data/s1_task_" + str(task_number) + "/" + "s1_task_" + str(task_number) + ".h5", 'w')
    
    hf.create_dataset('attributes', data=attributes)
    hf.create_dataset('boxes_1024', data=boxes_1024)
    hf.create_dataset('labels', data=labels)
    hf.create_dataset('img_to_first_box', data=img_to_first_box)
    hf.create_dataset('img_to_last_box', data=img_to_last_box)
    hf.create_dataset('img_to_first_rel', data=img_to_first_rel)
    hf.create_dataset('img_to_last_rel', data=img_to_last_rel)
    hf.create_dataset('predicates', data=predicate)
    hf.create_dataset('relationships', data=relationships_present)
    hf.create_dataset('split', data=split)
    
    hf.close()

def s1(filename, split, test_combined, exemp = False):

    #join filename to "VG-SGG-with-attri.h5"

    file_name = os.path.join(filename, "VG-SGG-with-attri.h5")

    f1 = h5py.File(file_name,'r+') 

    data_split = f1['split'][:]

    idx = np.where(data_split == 2)[0]

    f1_arr = np.array(f1)

    unique, counts = np.unique(data_split, return_counts=True)

    split_copy = data_split.copy()

    split_flag = 2 if split == 'test' else 0

    split_mask = data_split == split_flag

    init_len = len(np.where(split_mask)[0])

    split_mask &= f1['img_to_first_box'][:] >= 0

    num_val_im = 0

    split_mask &= f1['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]

    split_mask = np.zeros_like(data_split).astype(bool)

    if exemp == 10 :
        np.random.seed(0)
        np.random.shuffle(image_index)
        image_index = image_index[:int(len(image_index)*0.1)]


    split_mask[image_index] = True

    all_labels = f1['labels'][:, 0]

    all_attributes = f1['attributes'][:, :]

    all_boxes = f1['boxes_{}'.format(1024)][:]

    assert np.all(all_boxes[:, : 2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    im_to_first_box = f1['img_to_first_box'][split_mask]
    im_to_last_box = f1['img_to_last_box'][split_mask]
    im_to_first_rel = f1['img_to_first_rel'][split_mask]
    im_to_last_rel = f1['img_to_last_rel'][split_mask]

    _relations = f1['relationships'][:]
    _relation_predicates = f1['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check   

    dataset_dicts = {}

    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []

    for i in tqdm(range(len(image_index))):
    
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]
        
        
        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        # let the foreground start from 0
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1] - 1
        
        # the relationship foreground start from the 1, 0 for background
        gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]
        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            # (num_rel, 3), representing sub, obj, and pred
            rels = np.column_stack((obj_idx, predicates))
        else:
            rels = np.zeros((0, 3), dtype=np.int32)


        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)
    
    dataset_dicts["boxes"] =  boxes
    dataset_dicts["gt_classes"] = gt_classes
    dataset_dicts["gt_attributes"] = gt_attributes
    dataset_dicts["relationships"] = relationships
    dataset_dicts["split_mask"] = split_mask  

    task_distribution = {1: [31, 16, 25, 47, 19, 34, 37, 27, 39, 15],
                        2: [20, 11, 7, 46, 33, 28, 2, 3, 17, 18],
                        3: [48, 38, 23, 41, 6, 36, 26, 32, 42, 45],
                        4: [30, 22, 40, 49, 43, 44, 13, 10, 4, 12],
                        5: [29, 8, 50, 21, 1, 24, 14, 5, 9, 35]} 

    if test_combined or exemp:
        task_distribution[1] = task_distribution[1]
        task_distribution[2] = task_distribution[2] + task_distribution[1]
        task_distribution[3] = task_distribution[3] + task_distribution[2]
        task_distribution[4] = task_distribution[4] + task_distribution[3]
        task_distribution[5] = task_distribution[5] + task_distribution[4]
    
    print("Task distribution: ", task_distribution)
     
    divided_dataset_dict = divide_dataset(task_distribution,dataset_dicts)

    #Sanity check


    assert len(divided_dataset_dict[1]["boxes"]) == len(divided_dataset_dict[1]["gt_classes"])
    assert len(divided_dataset_dict[2]["boxes"]) == len(divided_dataset_dict[2]["gt_classes"])
    assert len(divided_dataset_dict[3]["boxes"]) == len(divided_dataset_dict[3]["gt_classes"])

    assert len(divided_dataset_dict[1]["boxes"]) == len(divided_dataset_dict[1]["gt_attributes"])
    assert len(divided_dataset_dict[2]["boxes"]) == len(divided_dataset_dict[2]["gt_attributes"])
    assert len(divided_dataset_dict[3]["boxes"]) == len(divided_dataset_dict[3]["gt_attributes"])

    assert len(divided_dataset_dict[1]["boxes"]) == len(divided_dataset_dict[1]["relationships"])
    assert len(divided_dataset_dict[2]["boxes"]) == len(divided_dataset_dict[2]["relationships"])
    assert len(divided_dataset_dict[3]["boxes"]) == len(divided_dataset_dict[3]["relationships"])

    assert len(divided_dataset_dict[1]["boxes"]) == sum(divided_dataset_dict[1]["split_mask"])
    assert len(divided_dataset_dict[2]["boxes"]) == sum(divided_dataset_dict[2]["split_mask"])
    assert len(divided_dataset_dict[3]["boxes"]) == sum(divided_dataset_dict[3]["split_mask"])

    assert len(divided_dataset_dict[1]["split_mask"]) == 108073
    assert len(divided_dataset_dict[2]["split_mask"]) == 108073
    assert len(divided_dataset_dict[3]["split_mask"]) == 108073

    num_tasks = 5 

    for i in range(num_tasks):
        if test_combined:
            split = "test combined"
            print("Creating Task " + str(i+1) + " " + split + " Dataset for S1!!!")
            convert_to_h5(divided_dataset_dict[i+1], i+1, "test", test_combined, exemp)
        else:
            print("Creating Task " + str(i+1) + " " + split + " Dataset for S1!!!")
            convert_to_h5(divided_dataset_dict[i+1], i+1, split, test_combined, exemp)
    
    f1.close()

def create_directory_structure_s1(root_path):

    image_data_path = os.path.join(root_path, "image_data.json")
    dict_attri_path = os.path.join(root_path, "VG-SGG-dicts-with-attri.json")

    print("Creating Directory Structure!!!")

    if not os.path.exists("s1_data/s1_task_1"):
        os.makedirs("s1_data/s1_task_1")
        os.makedirs("s1_data/s1_task_1_exemp_10")
        os.makedirs("s1_data/s1_task_1_exemp_100")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_1/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_1_exemp_10/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_1_exemp_100/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

    if not os.path.exists("s1_data/s1_task_2"):
        os.makedirs("s1_data/s1_task_2")
        os.makedirs("s1_data/s1_task_2_exemp_10")
        os.makedirs("s1_data/s1_task_2_exemp_100")
        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_2/")
        shutil.copyfile(image_data_path, target_path + "image_data.json")
        shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_2_exemp_10/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_2_exemp_100/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")
        
    if not os.path.exists("s1_data/s1_task_3"):
        os.makedirs("s1_data/s1_task_3")
        os.makedirs("s1_data/s1_task_3_exemp_10")
        os.makedirs("s1_data/s1_task_3_exemp_100")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_3/")
        shutil.copy(image_data_path, target_path + "image_data.json")
        shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_3_exemp_10/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_3_exemp_100/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

    if not os.path.exists("s1_data/s1_task_4"):
        os.makedirs("s1_data/s1_task_4")
        os.makedirs("s1_data/s1_task_4_exemp_10")
        os.makedirs("s1_data/s1_task_4_exemp_100")
        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_4/")
        shutil.copy(image_data_path, target_path + "image_data.json")
        shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_4_exemp_10/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_4_exemp_100/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

    if not os.path.exists("s1_data/s1_task_5"):
        os.makedirs("s1_data/s1_task_5")
        os.makedirs("s1_data/s1_task_5_exemp_10")
        os.makedirs("s1_data/s1_task_5_exemp_100")
        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_5/")
        shutil.copy(image_data_path, target_path + "image_data.json")
        shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_5_exemp_10/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")

        target_path = os.path.join(root_path[:-3], "s1_data/s1_task_5_exemp_100/")
        dest = shutil.copyfile(image_data_path, target_path + "image_data.json")
        dest = shutil.copy(dict_attri_path, target_path + "VG-SGG-dicts-with-attri.json")
    
def main(args):

    create_directory_structure_s1(args.file_path)
    print("Directory Structure Created!!!")
    # s1(args.file_path, split="train")
    # s1(args.file_path, split="test")
    #s1(args.file_path, split="test", test_combined=True)
    print("Creating Replay Buffer 10%")
    s1(args.file_path, split="train", test_combined=False, exemp=10)
    print("Creating Replay Buffer 100%")
    s1(args.file_path, split="train", test_combined=False, exemp=100)

    

    

#write a main function in python

if __name__ == "__main__":

    parser = default_argument_parser()
    parser.add_argument("--file_path", required=True)
    args = parser.parse_args()  

    main(args)



    
    


