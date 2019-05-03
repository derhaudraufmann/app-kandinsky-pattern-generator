import os
import sys

from PIL import Image
from kp import KandinskyUniverse, RandomKandinskyFigure, SimpleObjectAndShape, ShapeOnShapes, ShapeCombinations

amountCircles = 1
amountFigures = 50

if (__name__ == '__main__'):

    print('Welcome to the Kandinsky Figure Generator')

    if len(sys.argv) != 4:
        print("usage: mf_kpgen.py <function> <numberCircles> <numberFigures>")
        print("example: mf_kpgen.py circles 3 500")
        exit(1)
    else:
        amountCircles = int(sys.argv[2])
        amountFigures = int(sys.argv[3])

    u = KandinskyUniverse.RedUniverse()


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


# create 500 figures of true/false class each, for pattern: exactly 2 circles
redCirclesWithRandomOthers(amountCircles, amountFigures)
