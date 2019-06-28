import os
import sys
import threading

from PIL import Image
from kp import KandinskyUniverse, RandomKandinskyFigure, SimpleObjectAndShape, ShapeOnShapes, ShapeCombinations

amountCircles = 1
amountFiguresTrain = 100
amountFiguresTest = 30

if (__name__ == '__main__'):

    print('Welcome to the Kandinsky Figure Generator')

    # if len(sys.argv) != 5:
    #     print("usage: mf_kpgen.py <function> <numberCircles> <numberFiguresTrain> <numberFiguresTest>")
    #     print("example: mf_kpgen.py circles 3 1000 300")
    #     exit(1)
    # else:
    #     amountCircles = int(sys.argv[2])
    #     amountFiguresTrain = int(sys.argv[3])
    #     amountFiguresTest = int(sys.argv[4])




def twoSquaresOneRandom():
    # just playing around, trying to create a simple pattern
    os.makedirs("../test/TwoSquaresOneRandom/true", exist_ok=True)
    os.makedirs("../test/TwoSquaresOneRandom/false", exist_ok=True)

    twoSquaresObjects = ShapeCombinations.TwoSquaresOneRandom(u, 2, 2)
    print("the pattern is: ", twoSquaresObjects.humanDescription())

    kfs = twoSquaresObjects.true_kf(50)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../test/TwoSquaresOneRandom/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = twoSquaresObjects.false_kf(50)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../test/TwoSquaresOneRandom/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1


def redCirclesOnly(numberCircles, numberFiguresTrain=100, numberFiguresTest=30):
    os.makedirs("data/circlesOnly/" + str(numberCircles) + "/train/true", exist_ok=True)
    os.makedirs("data/circlesOnly/" + str(numberCircles) + "/train/false", exist_ok=True)
    os.makedirs("data/circlesOnly/" + str(numberCircles) + "/test/true", exist_ok=True)
    os.makedirs("data/circlesOnly/" + str(numberCircles) + "/test/false", exist_ok=True)

    circles = ShapeCombinations.CirclesOnly(u)
    print("the pattern is: ", circles.humanDescription())

# training set
    kfs = circles.true_kf(numberCircles, numberFiguresTrain)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/circlesOnly/" + str(numberCircles) + "/train/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.false_kf(numberCircles, numberFiguresTrain)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/circlesOnly/" + str(numberCircles) + "/train/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1

# test set
    kfs = circles.true_kf(numberCircles, numberFiguresTest)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/circlesOnly/" + str(numberCircles) + "/test/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.false_kf(numberCircles, numberFiguresTest)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/circlesOnly/" + str(numberCircles) + "/test/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1


def redCirclesRange(minNumberCircles, maxNumberCircles, numberFiguresTrain=100, numberFiguresTest=30):
    os.makedirs("data/CirclesRange/" + str(minNumberCircles) + "-" + str(maxNumberCircles) + "/train/true", exist_ok=True)
    os.makedirs("data/CirclesRange/" + str(minNumberCircles) + "-" + str(maxNumberCircles) + "/train/false", exist_ok=True)
    os.makedirs("data/CirclesRange/" + str(minNumberCircles) + "-" + str(maxNumberCircles) + "/test/true", exist_ok=True)
    os.makedirs("data/CirclesRange/" + str(minNumberCircles) + "-" + str(maxNumberCircles) + "/test/false", exist_ok=True)

    circles = ShapeCombinations.CirclesOnlyRange(u)
    print("the pattern is: ", circles.humanDescription())

# training set
    kfs = circles.true_kf(minNumberCircles, maxNumberCircles, numberFiguresTrain)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/CirclesRange/" + str(minNumberCircles) + "-" + str(maxNumberCircles) + "/train/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.false_kf(minNumberCircles, maxNumberCircles, numberFiguresTrain)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/CirclesRange/" + str(minNumberCircles) + "-" + str(maxNumberCircles) + "/train/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1

# test set
    kfs = circles.true_kf(minNumberCircles, maxNumberCircles, numberFiguresTest)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/CirclesRange/" + str(minNumberCircles) + "-" + str(maxNumberCircles) + "/test/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.false_kf(minNumberCircles, maxNumberCircles, numberFiguresTest)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/CirclesRange/" + str(minNumberCircles) + "-" + str(maxNumberCircles) + "/test/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1


def redCirclesWithRandomOthers(numberCircles, numberFigures=50):
    os.makedirs("data/circles/" + str(numberCircles) + "/true", exist_ok=True)
    os.makedirs("data/circles/" + str(numberCircles) + "/false", exist_ok=True)

    circles = ShapeCombinations.Circles(u)
    print("the pattern is: ", circles.humanDescription())

    kfs = circles.true_kf(numberCircles, numberFigures)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/circles/" + str(numberCircles) + "/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.false_kf(numberCircles, numberFigures)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "data/circles/" + str(numberCircles) + "/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1

def moreRedThanBlue(numberFiguresTrain=1000, numberFiguresTest=300):
    os.makedirs("../data/kandinsky/MoreRedThanBlue/train/true", exist_ok=True)
    os.makedirs("../data/kandinsky/MoreRedThanBlue/train/false", exist_ok=True)
    os.makedirs("../data/kandinsky/MoreRedThanBlue/test/true", exist_ok=True)
    os.makedirs("../data/kandinsky/MoreRedThanBlue/test/false", exist_ok=True)

    circles = ShapeCombinations.SetMoreRedThanBlue(u)
    print("the pattern is: ", circles.humanDescription())

    # training set
    print('Generating training set, %d samples', numberFiguresTrain)

    kfs = circles.true_kf(numberFiguresTrain)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/MoreRedThanBlue/train/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1


    kfs = circles.false_kf(numberFiguresTrain)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/MoreRedThanBlue/train/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    # test set

    print('Generating test set, %d samples', numberFiguresTest)

    kfs = circles.true_kf(numberFiguresTest)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/MoreRedThanBlue/test/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.false_kf(numberFiguresTest)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/MoreRedThanBlue/test/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1


def moreRedThanBlueSparse(numberFiguresTrain=1000, numberFiguresTest=300):
    os.makedirs("../data/kandinsky/MoreRedThanBlueSparse/train/true", exist_ok=True)
    os.makedirs("../data/kandinsky/MoreRedThanBlueSparse/train/false", exist_ok=True)
    os.makedirs("../data/kandinsky/MoreRedThanBlueSparse/test/true", exist_ok=True)
    os.makedirs("../data/kandinsky/MoreRedThanBlueSparse/test/false", exist_ok=True)

    circles = ShapeCombinations.SetMoreRedThanBlueSparse(u)
    print("the pattern is: ", circles.humanDescription())

    # training set
    print('Generating training set, %d samples', numberFiguresTrain)

    kfs = circles.train_true_kf(numberFiguresTrain)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/MoreRedThanBlueSparse/train/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.train_false_kf(numberFiguresTrain)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/MoreRedThanBlueSparse/train/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    # test set

    print('Generating test set, %d samples', numberFiguresTest)

    kfs = circles.test_true_kf(numberFiguresTest)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/MoreRedThanBlueSparse/test/true/%06d" % i
        image.save(filename + ".png")
        i = i + 1

    kfs = circles.test_false_kf(numberFiguresTest)
    i = 0
    for kf in kfs:
        image = KandinskyUniverse.kandinskyFigureAsImage(kf)
        filename = "../data/kandinsky/MoreRedThanBlueSparse/test/false/%06d" % i
        image.save(filename + ".png")
        i = i + 1


# create 500 figures of true/false class each, for pattern: exactly 2 circles
# redCirclesOnly(amountCircles, amountFiguresTrain)

u = KandinskyUniverse.AllColorCirclesUniverse()


# redCirclesRange(1, 3, amountFiguresTrain, amountFiguresTest)

# moreRedThanBlue(100, 30)

# moreRedThanBlueSparse(5000, 800)
