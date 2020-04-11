import random
import os
import numpy as np

def IOU(ann, centroids):
    '''Une fonction qui calcule le IoU (Intersection sur réunion) pour une donnée par rapport aux centroids (Anchors)'''

    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        # Calcul le meilleur IoU entre une donnée et une anchor
        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:
            similarity = (c_w*c_h)/(w*h)

        # La distance entre l'anchor et la donnée dans l'espace des valeurs des IoU
        similarities.append(similarity)

    return np.array(similarities)

def avg_IOU(anns, centroids):
    '''Unne fonction qui calcule la moyenne des IoU'''
    n,d = anns.shape
    som = 0.

    for i in range(anns.shape[0]):
        som+= max(IOU(anns[i], centroids))

    return som/n

def print_anchors(centroids):
    '''Une fonction qui renvoit les valeurs des anchors une fois le k-means est terminé'''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.2f,%0.2f, ' % (anchors[i,0], anchors[i,1])

    r += '%0.2f,%0.2f' % (anchors[sorted_indices[-1:],0], anchors[sorted_indices[-1:],1])
    r += "]"

    print(r)

def run_kmeans(ann_dims, anchor_num):
    '''L'algorithme k-means est utilisé pour trouvé un nombre n des anchors représant tout les données
       en minimisant les distance entre ces représantant et leurs représentés '''

    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("itération {}: distances = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def anchors_gen(anns_dir, num_anchors):
    '''Une fonction permettant de trouver les meilleurs n anchors représentants des données'''
    ind = 0
    train_images = []
    train_labels = {}

    # Commence par une lecture de toutes les données qu'on le fournit
    for ann in sorted(os.listdir(anns_dir)):
        with open(anns_dir + ann, mode = 'r') as file:
            count = 0
            for line in file:
                count += 1
                name, ymin, xmin, ymax, xmax, height, width = line[:-1].split(',')

                obj_ref = name + '_' + str(ind)

                train_images += [obj_ref]
                train_labels[obj_ref] = [int(height), int(width), int(ymin), int(xmin), int(ymax), int(xmax)] # 256,256,24,111,232,164

                ind += 1

            if count > max_objects:
                max_objects = count

    grid_w = MAX_DIM/19 # n'affecte pas beaucoup les résultats (car trés grands)
    grid_h = MAX_DIM/19 # mais c'est mieux d'utiliser 19 pour SqueezeNet et 39 pour MobileNet


    annotation_dims = []
    for obj_ref in train_images:
        height, width, ymin, xmin, ymax, xmax = train_labels[obj_ref]
        cell_w = width  / grid_w
        cell_h = height / grid_h

        relative_w = (xmax - xmin) / cell_w
        relative_h = (ymax - ymin) / cell_h

        annotation_dims.append((relative_w,relative_h))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    return centroids, annotation_dims
