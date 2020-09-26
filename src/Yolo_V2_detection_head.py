from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Reshape, Input, Activation, Conv2D, GlobalMaxPool2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.layers import ZeroPadding2D, Dropout, DepthwiseConv2D, LeakyReLU, ReLU, concatenate
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
import cv2
import os

from Yolo_V2_utils import *
from Yolo_V2_extractors import *
from Yolo_V2_preprocessing import *

seen, total_recall = None, None

class YOLO(object):
    '''Le model YOLO est un algorithme de détection basé sur les techniques d'apprentissage profond.
       Puisqu'il est toujours en cours de développement, en particulier sa tête de détection et sa fonction de coût,
       on implémente dans ce script la deuxième version YOLO_V2/YOLO9000 qui est plus stable en terme de gradient
       (utilisation des anchors) et plus rapide que celle qui la précède mais développée pour la détection
       d'objets multiples.'''

    def __init__(self,
                 backend,
                 input_size,
                 labels,
                 max_box_per_image,
                 anchors):
        '''La fonction/méthode d'initialisation du model YOLO'''
        
#         super(YOLO, self).__init__()
        self.input_size = input_size
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box   = len(anchors)//2
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors  = anchors
        self.max_box_per_image = max_box_per_image

        ################################################
        # Création du model : graphe de computation !! #
        ################################################

        # Vecteur d'entrée qui va contenir l'image : On peut utiliser une seul canal pour le grey mode
        input_image     = Input(shape=(self.input_size, self.input_size, 3))

        # On définit le vecteur de sortie/entrée pour ne pas défiir deux structures.
        # La structure d'entrainement qui nécessite deux entrées : image et valeurs réelles
        # La structure de prédiction qui nécessite seulement une image en entrée
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))

        # Choisir l'extracteur de caractéristiques
        if backend == 'DarkNet':
            self.feature_extractor = DarkNetFeature(self.input_size)
        elif backend == 'SqueezeNet':
            self.feature_extractor = SqueezeNetFeature(self.input_size)
        elif backend == 'MobileNet':
            self.feature_extractor = MobileNetFeature(self.input_size)
        else:
            raise Exception('Architecture n\'est pas supportée!')

        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape() # Définition de la taille des grilles en fonction de la sortie de l'extracteur
        features = self.feature_extractor.extract(input_image) # Définir le Vecteur de caractéristiques

        # Créer la dérnière couche de détection
        # 4 coordonnées, 1 prob d'existence, C classes d'objets, A anchors
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class), 
                        (1,1), strides=(1,1),
                        padding='same',
                        name='Couche_de_detection',
                        kernel_initializer='lecun_normal')(features)

        # Transformer le vecteur de sortie en tenseur de la forme finale
        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class), name = "reshape")(output)
        output = Lambda(lambda args: args[0], name = "Lambda")([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output, name = "Yolo")

        ######################################################
        # initialisation des poids de la couche de détection #
        ######################################################

        # Extrairaction de la dérnière couche de l'extracteur de caractéristiques
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        # Normalisation des poids de cette couche
        new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)

        layer.set_weights([new_kernel, new_bias])

        # Visualiser la structure du model final
        self.model.summary()

    def custom_loss(self, y_true, y_pred):
        '''Une implémentation de la fonction de coût comme décrite par Joseph Redmon pour Yolo_V2.
           Les nouveauté dans cette implémentation est l'utilisation de la détection par anchors.
           L'annotation [..., x:y] est utilisée puisque la fonction est définie sur des batches.'''

        #######################################################################################
        # Définition des variables : Les tenseurs sont vides dans un graphe de computation !! #
        #######################################################################################

        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)), tf.float32)
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0., trainable=False) # Compteur des élements traités du batche
        total_recall = tf.Variable(0., trainable=False) # Erreur/Coût final

        #######################################################
        # Ajustement et définition des variables de prédiction#
        #######################################################

        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid # Ajustement de x et y : entre 0 et 1 plus offset
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.nb_box,2]) # Ajustement de w et h
        pred_box_conf = tf.sigmoid(y_pred[..., 4]) # Ajustement de la fidélité (Proba de présence)
        pred_box_class = y_pred[..., 5:] # Ajustement des probabilités des classes

        ##################################################
        # Ajustement et définition des variables réelles #
        ##################################################

        # Positon et dimensions
        true_box_xy = y_true[..., 0:2] # Position relative à la grille
        true_box_wh = y_true[..., 2:4] # Dimensions relative à la grille

        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half # x_min et y_min réels
        true_maxes   = true_box_xy + true_wh_half # x_max et y_max réels

        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half # x_min et y_min prédits
        pred_maxes   = pred_box_xy + pred_wh_half # x_max et y_max prédits



        # Calcul de l'insection entre réel et prédit
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        # Calcul de la réunion entre réel et prédit
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        union_areas = pred_areas + true_areas - intersect_areas

        # Calcul de IoU : Intersection sur réunion entre réel et prédit
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        # On définit la fidélité/confiance de la prédiction comme produit de IoU et la présence
        # Par suite, la minimasation de la fonction de coût revient à maximiser IoU pour les grilles
        # "responsables pour la détection" et minimiser la présence pour les autres grilles.
        true_box_conf = iou_scores * y_true[..., 4]

        # Probabillités des classes : On prend le maximum des probabilités de classes,
        # donc on ne peut définir qu'un seul objet pour chaque grille.
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        ##############################################################################
        # Définition des masques : les éléments de la détection qui seront pénalisés #
        ##############################################################################

        # Un masque pour les prédicteurs des coordonées
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        # Un masque pour les prédicteurs qui ont un IoU inférieur à 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half

        # Calcul de l'insection entre réel et prédit
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        # Calcul de la réunion entre réel et prédit
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        union_areas = pred_areas + true_areas - intersect_areas

        # Calcul de IoU : Intersection sur réunion entre réel et prédit
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        # Les prédicteurs de fidélité/confidence/sûreté à pénaliser avec les poids de pénalisation
        # Celui qui sait ne parle pas, celui qui parle ne sait - Lao-Tseu
        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.cast(best_ious < 0.6, tf.float32) * (1 - y_true[..., 4]) * self.no_object_scale # Celui qui ne doit pas parler
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale # Celui qui doit savoir

        # Les prédicteurs des classes qui seront pénalisés avec les poids de pénalisation
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale

        # Les prédicteurs des anchors des grilles qui seront pénalisés : Ceux qui ne doivent pas détecter cet objet
        no_boxes_mask = tf.cast(coord_mask < self.coord_scale/2., tf.float32)

        # Incrémentation du compteur de batches
        seen = tf.compat.v1.assign_add(seen, 1.)

        # Le calcul pendant l'échauffement se fait différemment,
        # donc on introduit une fonction de conditionnement
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches+1), 
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                       true_box_wh + tf.ones_like(true_box_wh) * \
                                                       np.reshape(self.anchors, [1,1,1,self.nb_box,2]) * \
                                                       no_boxes_mask,
                                                       tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy, true_box_wh, coord_mask])


        ######################################
        # Finalisatin de la fonction du coût #
        ######################################

        # Calcul du nombre des erreurs prises en consedération.
        nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32))
        nb_conf_box  = tf.reduce_sum(tf.cast(conf_mask  > 0.0, tf.float32))
        nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32))

        # Calcul final des coûts de régression : on utilise la moyenne des erreurs carrées pondérées (par les pénalités choisies)
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.

        # Calcul final des coût de classification : on utilise l'entropie croisée entre chaque réel et prédit et on moyenne le tout
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = tf.cond(tf.less(seen, self.warmup_batches+1),
                      lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                      lambda: loss_xy + loss_wh + loss_conf + loss_class)

        ##############################################
        # Visualisation du progrés de l'entrainement #
        ##############################################

        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.cast(true_box_conf > 0.5, tf.float32) * tf.cast(pred_box_conf > 0.3, tf.float32))

            current_recall = nb_pred_box/(nb_true_box + 1e-6)
            total_recall = tf.compat.v1.assign_add(total_recall, current_recall)

            loss = tf.compat.v1.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            loss = tf.compat.v1.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)

        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def train(self,
              train_imgs,                  # Les images sur lesquelles le model sera entrainé
              valid_imgs,                  # Les images sur lesquelles la performance du model sera validé
              train_times,                 # Le nombre de fois pour répéter l'entrainement
              valid_times,                 # Le nombre de fois pourrépéter la validation
              nb_epochs,                   # Le nombre d'époques
              learning_rate,               # Le taux d'apprentissage initial : \ethat_{0}
              batch_size,                  # La taille du batch : Puisqu'on utilise un entrainement par batch
              warmup_epochs,               # Nombre d'époques pour l'échauffement
              object_scale,                # Pénalité sur les fausse négative : un objet n'est pas détecté
              no_object_scale,             # Pénalité sur les fausse positive : une détection d'objets inexistant
              coord_scale,                 # Pénalité sur les coordonnées du centre de l'objet
              class_scale,                 # Pénalité sur les dimensions de l'objet détecté
              saved_weights_dir,           # Chemin d'enregistrement du model
              logs_dir,
              debug=False                  # Activer/Désactiver la visualisation
              ):

        self.batch_size      = batch_size
        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale
        self.debug           = debug

        ########################################
        # Création des générateurs des données #
        ########################################

        # Configuretion des générateurs de données
        generator_config = {
            'IMAGE_H'         : self.input_size,
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }

        train_generator = BatchGenerator(train_imgs,
                                         generator_config,
                                         norm=self.feature_extractor.normalize)
        valid_generator = BatchGenerator(valid_imgs,
                                         generator_config,
                                         norm=self.feature_extractor.normalize,
                                         jitter=False)

        self.warmup_batches  = warmup_epochs * (train_times*len(train_generator) + valid_times*len(valid_generator))

        #############################################################
        # Choix d'algorithme d'optimisation et compilation du model #
        #############################################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################
        # Création des "Callbacks" #
        ############################

        # EarlyStopping pour l'arrêt d'entrainement
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   patience=3,
                                   mode='min',
                                   verbose=1)

        # ModelCheckpoint pour le sauvegarde des models
        file_path = saved_weights_dir + "EPOCH_{epoch:02d}.h5"
        checkpoint = ModelCheckpoint(file_path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     save_freq='epoch')

        # Tensorboard pour la visualisation
        tensorboard = TensorBoard(log_dir=logs_dir,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=False)

        ############################
        # Commencer l'entrainement #
        ############################

        # Entrainement sur les générateurs des données
        self.model.fit_generator(generator        = train_generator,
                                 steps_per_epoch  = len(train_generator) * train_times,
                                 epochs           = warmup_epochs + nb_epochs,
                                 verbose          = 1 if debug else 0,
                                 validation_data  = valid_generator,
                                 validation_freq  = 1,
                                 validation_steps = len(valid_generator) * valid_times,
                                 callbacks        = [tensorboard, checkpoint, early_stop],
                                 workers          = 3,
                                 max_queue_size   = 8)

    def predict(self, image):
        '''Fonction de prédiction des coordonnées de l'objet dans une image'''

        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size)) # Redimensionnement de l'image pour l'adapter à l'entrée du model
        image = self.feature_extractor.normalize(image) # Normalisation de l'image d'entrée our l'extracteur

        input_image = image[:,:,::-1] # Inversion des canaux de l'image
        input_image = np.expand_dims(input_image, 0) # Ajouter un nouvelle dimension à l'image
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4)) # Le tenseur utilisé en entrainement

        netout = self.model.predict([input_image, dummy_array]) # Prédiction du model
        boxes  = decode_netout(netout, self.anchors, self.nb_class) # Décodage de la sortie du model

        return boxes
