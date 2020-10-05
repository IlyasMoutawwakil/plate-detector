import os
import cv2
import copy
import sys
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
from Yolo_V2_utils import *

def parse_annotation(data_dir, datasets, labels=[]):
    '''Une fonction permettant de regrouper toutes les données en une seule variable'''
    
    img_idx = 0
    all_imgs = []
    seen_labels = {}
    for dataset in datasets :
        imgs_dir = data_dir + dataset + '/imgs/'
        anns_dir = data_dir + dataset + '/anns/'
        imgs_num = len(os.listdir(anns_dir))
        print(imgs_num)
        for img_ in sorted(os.listdir(imgs_dir)):
            img = {'object' : []} # img = {'object' : [obj], 'filename' : , 'height' : , 'width' : }

            img_name = imgs_dir + img_
            ann_name = anns_dir + img_[:-3] + 'txt'

            img['file_name'] = img_name

            with open(ann_name, mode = 'r') as file:
                for line in file:
                    obj = {'name' : ''} # obj = {'name' : '', 'ymin' : , 'xmin' : , 'ymax' : , 'xmax' : }
                    name, ymin, xmin, ymax, xmax, height, width = line[:-1].split(',')
                    obj['name'] = name
                    obj['ymin'], obj['xmin'], obj['ymax'], obj['xmax'] = int(ymin), int(xmin), int(ymax), int(xmax)
                    img['height'], img['width'] = int(height), int(width)
                    img['object'] += [obj]

            if obj['name'] in seen_labels:
                seen_labels[obj['name']] += 1
            else:
                seen_labels[obj['name']] = 1

            all_imgs += [img]
            
            prog = (img_idx / imgs_num) * 100
            if int(prog) == prog:
                sys.stdout.write("\r%d%%" %prog)
                sys.stdout.flush()
            
            img_idx += 1
            
    return all_imgs, seen_labels

class BatchGenerator(Sequence):
    '''Un objet générateur de batches (une dizaine d'éléments de la base des données) avec augmentation des données (Transformations de forme/couleur/...)'''

    def __init__(self, images,
                       config,
                       shuffle=True,
                       jitter=True,
                       norm=None):

        self.generator = None
        self.images = images
        self.config = config
        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm
        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

        ########################################################################################
        # Création de l'augmentateur des données : Applique des transformations sur les images #
        ########################################################################################

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.aug_pipe = iaa.Sequential(
                    [sometimes(iaa.Affine()), iaa.SomeOf((0, 5),
                    [iaa.OneOf([iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                                ]),
                     iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                     iaa.OneOf([iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                                ]),
                     iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                     iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                     iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                     ], random_order=True)], random_order=True)

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['file_name'])

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))
        for train_instance in self.images[l_bound:r_bound]:

            # Auglenter l'image (Appliquer les transformations sur les données)
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)

            # Construction du tenseur de sortie pour une image
            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])

                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell

                        box = [center_x, center_y, center_w, center_h]

                        # Choisir l'anchor qui permet la meilleur détection
                        best_anchor = -1
                        max_iou     = -1

                        shifted_box = BoundBox(0,
                                               0,
                                               center_w,
                                               center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou

                        # Définir le vecteur corrspondant à la gille contenant le centre de l'objet
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1

                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

            # assign input image to x_batch
            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'],
                                    (obj['xmin']+2, obj['ymin']+12),
                                    0, 1.2e-3 * img.shape[0],
                                    (0,255,0), 2)

                x_batch[instance_count] = img

            # Augmenter le compteur des instances pour le batche en cours
            instance_count += 1

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['file_name']
        image = cv2.imread(image_name)

        if image is None: print('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            # scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            # translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy : (offy + h), offx : (offx + w)]

            # flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)

            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin

        return image, all_objs
