import tensorflow as tf
import numpy as np

class BoundBox:
    ''' Un objet contenant les des informations sur un objet dans une image '''

    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

class WeightReader:
    ''' Lecteur des poids à partir du fichier h5 '''

    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4

def bbox_iou(box1, box2):
    ''' Une fonction qui calcule le quotient de l'intersection sur la réunion entre de rectangle '''

    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union

def draw_boxes(image, boxes, labels):
    '''Une fonction pour dessiner les rectangle entourant une plaque et et écrire son nom'''

    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 5)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score())[:4],
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2e-3 * image_h,
                    (0,255,0), 5)

    return image

def get_plates(image, boxes):
    '''Une fonction pour retourner seulement les RoI : régions d'intérêt'''
    image_h, image_w, _ = image.shape

    plates = []
    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        plates += [image[xmin:xmax, ymin:ymax, :]]

    return plates

def decode_netout(netout, anchors, nb_class, obj_threshold, nms_threshold):
    '''Une fonction pour décoder la sortie du réseau de neurones'''

    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # On décode la sortie tout les vecteur des grilles
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    # Parcours de tout les grilles et les objets qu'elles peuvent détecter
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # Les derniers éléments de classification et de présence/fidélité
                classes = netout[row,col,b,5:]

                if np.sum(classes) > 0:
                    # Les quatre premiers éléments de régression : coordonnées et dimensions
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + _sigmoid(x)) / grid_w # position x de l'objet relative à l'image
                    y = (row + _sigmoid(y)) / grid_h # position y de l'objet relative à l'image
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # largeur de l'objet relative à l'image
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # longueur de l'objet relative à l'image
                    confidence = netout[row,col,b,4] # La confidence/sûreté/fidélité

                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    boxes.append(box)

    # La suppresion non maximale :
    # Supprimer les détections qui ont une grande IoU : car ils seront détection du même objet
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            # Si la détection affirme que ce n'est pas un objet, on fait rien.
            if boxes[index_i].classes[c] == 0:
                continue

            # Si la détection affirme que c'est un objet, on compare sa IoU avec les autres objets
            else:
                # L'élimination se fait en laisantr celui avec la plus grande confidence/suûreté/fidélité
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    # Et on le juge non-objet s'il a une IoU élevée
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # Supprimer les détections qui ont une faible confidence/suûreté/fidélité
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes

def compute_overlap(a, b):
    """Une fonction qui calcul la valeur IoU : Intersection sur réunion"""

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _interval_overlap(interval_a, interval_b):
    '''Une fonction qui calcule la longueur d'intersection entre deux intervalle'''

    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def _sigmoid(x):
    '''La fonction sigmoid en utilisant np puisque tensorflow ne permet pas les calculs direct : Eager execution'''

    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    '''La fonction softmax en utilisant np puisque tensorflow ne permet pas les calculs direct : Eager execution'''

    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)
