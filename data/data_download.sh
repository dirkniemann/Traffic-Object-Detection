#!/bin/bash

# Paralles data download

wget -c https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.tar &    # Audi A2D2 dataset for semantic segmentation
wget -c https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip &     # BDD dataset Labels
wget -c https://dl.cv.ethz.ch/bdd100k/data/100k_images_test.zip &   # BDD Dataset test images
wget -c https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip &  # BDD Dataset train images
wget -c https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip &    # BDD Dataset val images
wget -c https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip & # GTSDB Dataset

# waiting for data download
wait

echo "Data download successfully!"