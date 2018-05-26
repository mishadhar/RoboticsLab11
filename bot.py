#!/usr/bin/env python3

'''Make Cozmo behave like a Braitenberg machine with virtual light sensors and wheels as actuators.

The following is the starter code for lab.
'''

import asyncio
import time
import cozmo
import cv2
import numpy as np
import sys
from imgclassification import *


def sense_brightness(image, columns):
    '''Maps a sensor reading to a wheel motor command'''
    ## TODO: Test that this function works and decide on the number of columns to use

    h = image.shape[0]
    w = image.shape[1]
    avg_brightness = 0

    for y in range(0, h):
        for x in columns:
            avg_brightness += image[y, x]

    avg_brightness /= (h * columns.shape[0])

    return avg_brightness


def mapping_funtion(sensor_value):
    '''Maps a sensor reading to a wheel motor command'''
    ## TODO: Define the mapping to obtain different behaviors.
    motor_value = 0.1 * sensor_value
    return motor_value


async def bot(robot: cozmo.robot.Robot):
    '''The core of the braitenberg machine program'''
    # Move lift down and tilt the head up
    robot.move_lift(-3)
    robot.set_head_angle(cozmo.robot.MAX_HEAD_ANGLE).wait_for_completed()
    print("Press CTRL-C to quit")

    img_clf = ImageClassifier()

    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    (test_raw, test_labels) = img_clf.load_data_from_folder('./test/')

    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    test_data = img_clf.extract_image_features(test_raw)

    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)
    predicted_labels = img_clf.predict_labels(train_data)
    print("\nTraining results")
    print("=============================")
    print("Confusion Matrix:\n", metrics.confusion_matrix(train_labels, predicted_labels))
    print("Accuracy: ", metrics.accuracy_score(train_labels, predicted_labels))
    print("F1 score: ", metrics.f1_score(train_labels, predicted_labels, average='micro'))

    cache = []
    idcount = 0
    lastImg = None

    while True:
        # get camera image
        event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

        # convert camera image to opencv format
        opencv_image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_RGB2GRAY)

        process_image = img_clf.extract_image_features([opencv_image])

        curimg = img_clf.predict_labels(process_image)[0]

        if len(cache) > 0 and len(cache) == 10:
            lastImg = cache.pop(0)

        cache.append(curimg)

        for i in cache:
            if(lastImg is not None):
                if i == curimg and i != 'none':
                    idcount = idcount + 1

        print(cache)

        if idcount > 7:
            print(curimg)
            await robot.say_text(curimg).wait_for_completed()
            robot.move_lift(10)
            time.sleep(4)
            idcount = 0
            cache = []
            robot.move_lift(-3)



cozmo.run_program(bot, use_viewer=True, force_viewer_on_top=True)
