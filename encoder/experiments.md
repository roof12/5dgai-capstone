# Experiments

## v1.0.0

First simple CNN, train on 1 mil, test on 100k

* early exit, somewhere around epoch 20
* Test Loss: 0.0552
* Test Accuracy: 0.9785

## v2.0.0

Change model, adding BatchNormalization and Activation after each Conv2D

* early exit, best epoch 30
* Test Loss: 0.0254
* Test Accuracy: 0.9922

## v2.0.1

Change data to use random mainline position instead of end position

* ???

## v3.0.0

Add outputs for castling rights

* early exit, unknown epoch
* Test loss: 0.1565
* Test output_material_cmp accuracy: 0.9970
* Test castle_white_kingside accuracy: 0.9963
* Test castle_white_queenside accuracy: 0.9884
* Test castle_black_kingside accuracy: 0.9969
* Test castle_black_queenside accuracy: 0.9917
