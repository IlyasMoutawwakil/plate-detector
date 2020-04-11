import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import shutil

# Définition des constantes du programme
DATA_DIR = 'E:\\ALPR\\TIPE\\data\\MOROCCAN_DATASET_RESIZED\\'
IMGS_DIR = DATA_DIR + 'imgs\\'
ANNS_DIR = DATA_DIR + 'anns\\'
OBJECT = 'Plate'

def write_ann_file(img_name, boxes, object = OBJECT):
    '''Une fontion qui enregistre les coordonnées en plus d'autres données dans un fichier txt'''

    # On définit la variable data pour ne pas avoir des exceptions si aucun objet n'est annoté
    data = ''
    for elm in boxes:
        top, left, bottom, right = min(boxes[elm][1],boxes[elm][3]), min(boxes[elm][0],boxes[elm][2]), max(boxes[elm][1],boxes[elm][3]), max(boxes[elm][0],boxes[elm][2])
        data += '{},{},{},{},{},{},{}\n'.format(object, top, left, bottom, right, height, width)

    with open(ANNS_DIR + img_name[:-4] + '.txt', mode = 'w') as ann_file:
        ann_file.write(data)
        print(data)

def update_coord(event,x,y,flags,param):
    '''Une fonction qui permet l'annotation et la visualisation du processus'''

    global img, bbox_x, bbox_y, draw, boxes, box_number

    # L'évènement d'appui sur le le bouton gauche
    if event == cv2.EVENT_LBUTTONDOWN :
        box_number += 1
        draw = True
        bbox_x = [x]
        bbox_y = [y]
        cv2.rectangle(img, (bbox_x[0], bbox_y[0]), (bbox_x[-1], bbox_y[-1]), (0, 255, 0), 2, lineType = 0)

    # L'évènement de bouger la souris tout en appuyan sur le bouton gauche
    elif event == cv2.EVENT_MOUSEMOVE and draw:
        img = cv2.imread(IMGS_DIR + img_name)
        for elm in boxes:
            cv2.rectangle(img, (boxes[elm][0], boxes[elm][1]), (boxes[elm][2], boxes[elm][3]), (0, 255, 0), 2, lineType = 0)
        bbox_x += [x]
        bbox_y += [y]
        cv2.rectangle(img, (bbox_x[0], bbox_y[0]), (bbox_x[-1], bbox_y[-1]), (0, 255, 0), 2, lineType = 0)

    # L'évènement de lâcher le bouton gauche
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        bbox_x += [x]
        bbox_y += [y]
        cv2.rectangle(img, (bbox_x[0], bbox_y[0]), (bbox_x[-1], bbox_y[-1]), (0, 255, 0), 2, lineType = 0)
        boxes[box_number] = bbox_x[0], bbox_y[0], bbox_x[-1], bbox_y[-1]

    cv2.imshow(win_name, img)

# Programme principal d'annotation
# Les commande sont : n pour next, q pour quit, r pour redo
# Les commande peuvent être implémentées à l'aide des boutons mais j'ai pas un PC pour le moment

if __name__ == '__main__':

    # Définition des variables globales du programme
    img = None
    draw = False

    for img_name in os.listdir(IMGS_DIR):

        win_name = 'Annotation Tool' # Nom de la fenêtre
        cv2.namedWindow(win_name) # Nommer la fenêtre
        cv2.setMouseCallback(win_name, update_coord) # Mettre à jour la fonction update_coord avec les coordonnées et les évènement de la souris
        img = cv2.imread(IMGS_DIR + img_name) # L'image d'initialisation
        height, width, channels = img.shape

        # Initialisation des variables globales
        bbox_x, bbox_y = [], []
        boxes = {}
        box_number = 0

        # Boucle d'annotation
        while 1:
            cv2.imshow(win_name, img) # Visualisation de l'image mise à jour à chaque itération
            action = cv2.waitKey()

            if (action == ord('n') and len(bbox_x) > 0) or action == ord('q'):
                break

            elif action == ord('r'):
                box_number = 0
                boxes = {}
                bbox_x, bbox_y = [], []
                img = cv2.imread(IMGS_DIR + img_name)

        if action == ord('q'):
            break
        write_ann_file(img_name, boxes)

cv2.destroyAllWindows()

## Programme principal de vérification : permet de définir une liste d'image à revoir
#to_redo = []
#for img_name in os.listdir(IMGS_DIR):
#    if img_name < to_redo[-1] :
#        continue
#    img = cv2.imread(IMGS_DIR + img_name)
#    cv2.imshow('img',img)
#    action = cv2.waitKey()
#    if action == ord('q'):
#        break
#    elif action == ord('r'):
#        to_redo += [img_name]
#    else:
#        continue
