#!/bin/bash

# unzip all datasets and remove the .zip and .tar data

unzip gtCoarse.zip
rm gtCoarse.zip
unzip 100k_images_test.zip
rm 100k_images_test.zip
unzip 100k_images_train.zip
rm 100k_images_train.zip
unzip 100k_images_val.zip
rm 100k_images_val.zip
unzip bdd100k_det_20_labels_trainval.zip
rm bdd100k_det_20_labels_trainval.zip
unzip FullIJCNN2013.zip
rm FullIJCNN2013.zip
tar xfv camera_lidar_semantic.tar
rm camera_lidar_semantic.tar

# renaming the datasets
mv camera_lidar_semantic audi_a2d2
mv FullIJCNN2013 traffic_signs

