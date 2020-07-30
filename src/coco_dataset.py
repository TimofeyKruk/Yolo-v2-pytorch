"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from torch.utils.data import Dataset
from src.data_augmentation import *
import pickle
import copy


class COCODataset(Dataset):
    def __init__(self, root_path="data//COCO", year="2014", mode="train", image_size=448, is_training=True):
        if mode in ["train", "val"] and year in ["2014", "2015", "2017"]:
            self.image_path = os.path.join("//media//cuda//HDD//Internship//Kruk//COCO//", "images",
                                           "{}{}".format(mode, year))
            anno_path = os.path.join("data//COCO", "anno_pickle", "COCO_{}{}.pkl".format(mode, year))
            id_list_path = pickle.load(open(anno_path, "rb"))

        self.classes = ["person", "car", "bird", "cat", "dog"]
        self.class_ids = [1, 3, 16, 17, 18]

        humans_already_added = 0

        humans_thresh = 10000

        new_list = []
        for dict in id_list_path:
            new_objects = []
            only_humans = True
            for object in id_list_path[dict]["objects"]:
                if object[4] in self.class_ids:
                    new_objects.append(object)
                    if object[4] != 1:  # not human
                        only_humans = False

            if len(new_objects) != 0:
                if only_humans is True and humans_already_added < humans_thresh:
                    new_list.append({"file_name": id_list_path[dict]["file_name"],
                                     "objects": new_objects})
                    humans_already_added += 1
                else:
                    if only_humans is False:
                        new_list.append({"file_name": id_list_path[dict]["file_name"],
                                         "objects": new_objects})

        # self.id_list_path = list(id_list_path.values())
        self.id_list_path = new_list

        # self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        #                 "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        #                 "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        #                 "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        #                 "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        #                 "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        #                 "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        #                 "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        #                 "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        #                 "teddy bear", "hair drier", "toothbrush"]
        # self.class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
        #                   31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        #                   55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        #                   82, 84, 85, 86, 87, 88, 89, 90]
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.id_list_path)
        print("##### Num images coco_dataset: ", self.num_images)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = os.path.join(self.image_path, self.id_list_path[item]["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        objects = copy.deepcopy(self.id_list_path[item]["objects"])
        for idx in range(len(objects)):
            objects[idx][4] = self.class_ids.index(objects[idx][4])
        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])

        image, objects = transformations((image, objects))
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
