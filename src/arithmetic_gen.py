import os
import sys
import threading

from PIL import Image
from kp import KandinskyUniverse, RandomKandinskyFigure, SimpleObjectAndShape, ShapeOnShapes, ShapeCombinations


# generate test and training data where #red circles plus #blue circles equals #yellow circles

# @input integer numberFiguresTrain: number of training figures per class to be generated
# @input integer numberFiguresTest: number of test figures per class to be generated
# @input integer train_offset [optional]: offset for naming files in training set, can be used for generating into the same data
#   folder sequentially or concurrently
# @input integer train_offset [optional]: offset for naming files in test set, can be used for generating into the same data
#   folder sequentially or concurrently
def redPlusBlueEqualsYellow(numberFiguresTrain=1000, numberFiguresTest=300, train_offset=0, test_offset=0):
    os.makedirs("../data/kandinsky/RedPlBlueIsYell_big/train/true", exist_ok=True)
    os.makedirs("../data/kandinsky/RedPlBlueIsYell_big/train/false", exist_ok=True)
    os.makedirs("../data/kandinsky/RedPlBlueIsYell_big/test/true", exist_ok=True)
    os.makedirs("../data/kandinsky/RedPlBlueIsYell_big/test/false", exist_ok=True)

    circles = ShapeCombinations.ArithRplusBisY(u)
    print("the pattern is: ", circles.humanDescription())

    # training set
    print('Generating training set, %d samples', numberFiguresTrain)

    kfs = circles.train_true_kf(numberFiguresTrain)
    i = train_offset
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/RedPlBlueIsYell_big/train/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.train_false_kf(numberFiguresTrain)
    i = train_offset
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/RedPlBlueIsYell_big/train/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    # test set

    print('Generating test set, %d samples', numberFiguresTest)

    kfs = circles.test_true_kf(numberFiguresTest)
    i = test_offset
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/RedPlBlueIsYell_big/test/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.test_false_kf(numberFiguresTest)
    i = test_offset
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/RedPlBlueIsYell_big/test/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1


redPlusBlueEqualsYellow(30000, 9000)
