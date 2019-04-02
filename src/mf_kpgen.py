import os

from PIL import Image
from kp import KandinskyUniverse, RandomKandinskyFigure, SimpleObjectAndShape, ShapeOnShapes, ShapeCombinations

if (__name__ == '__main__'):

    print('Welcome to the Kandinsky Figure Generator')
    u = KandinskyUniverse.SimpleUniverse()

    randomKFgenerator = RandomKandinskyFigure.Random(u, 4, 4)
    kfs = randomKFgenerator.true_kf(50)

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
