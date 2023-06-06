# coding=utf-8
# callbacks.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu and Pablo Doñate.
#
# Using Machine Learning techniques for image enhancement.
# This file has been created by jlaihong and modified by Pablo Doñate Navarro (800710@unizar.es).

import tensorflow as tf

class SaveCustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager, steps_per_epoch):
        self.checkpoint_manager = checkpoint_manager
        self.steps_per_epoch = steps_per_epoch
    

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint_manager.checkpoint.epoch.assign_add(1)
        self.checkpoint_manager.checkpoint.step.assign_add(self.steps_per_epoch)
        self.checkpoint_manager.save()