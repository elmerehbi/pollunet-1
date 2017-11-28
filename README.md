# Introduction
pollunet: https://github.com/hermenemacs/pollunet

pix2pix: https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix

U-net: https://arxiv.org/pdf/1505.04597.pdf


Ce dépôt contient une adaptation du réseau de neurones U-net, pour une expérimentation sur la détection de pollutions maritimes sur des images satellites (SAR).
Le jeu de données a été fourni par l'entreprise CLS (www.cls.fr/).

# Requirement

h5py, keras, matplotlib, numpy, tensorflow-gpu et leur dépendances.

# Fonctionnement du code

## Modèles

Chaque modèle de réseau de neurones est dans un fichier différent, qui contient la structure du modèle, le code pour prétraiter les données dont il a besoin, entraîner le réseau et l'appliquer sur les jeux de données de test.
Pour des raisons de simplicité, seul le fichier gmf3.py est commenté (c'est le plus récent, et le plus propre), mais tous les autre fichiers ont une structure similaire.

## Jeu de données

Le jeu de données de base consiste en 204 fichiers netcdf contenant une image SAR de dimensions variables prise autour du détroit de Gibraltar, la segmentation correspondante, les données de vent déduites de l'image et prédite, l'angle d'incidence selon une dimension, et des données de fréquence de pollution et  baythymétrie, qui n'ont pas été utilisées ici.

Il a d'abord été séparé en deux dossiers, train et test, test contenant les 34 images les plus récentes (octobre 2015 à mars 2017) et train les autres (jusqu'à mai 2017). Ces deux dossiers séparent les images d'entraînement des images de test. Des patches en ont ensuite été extraits (img_extract.py).

Sur pc-sc-086, les données se présentent sous la forme suivante:
```
|__Database_netcdf: netcdf_dir dans le code
|  |__train: 170 fichiers pour l'entraînement
|  |__test: 34 fichier pour les test
|__data.hdf5: fichier hdf contenant tout (patch extraits, masques, poids, résultats), correspondant à la variable hdf dans le code
|__weights: dossiers contenant les poids des réseaux entraînés, par nom de fichier correspondant (weight_dir dans le code)
   |__4classes: poids des réseaux entraînés pour reconnaître 4 classes, ensuite modifiés pour 3 classes (commit 'retiré la terre')
```




