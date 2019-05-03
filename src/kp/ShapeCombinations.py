import PIL
import random

from .KandinskyTruth import KandinskyTruthInterfce
from .KandinskyUniverse import kandinskyShape, overlaps
from .RandomKandinskyFigure import Random


class TwoSquaresOneRandom(KandinskyTruthInterfce):

    def isfuzzy(self):
        return false

    def humanDescription(self):
        return "contain two squares plus one random shape(square, triangle or circle)"

    def true_kf(self, n=1):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 3, 3)
        while i < n:
            kf = randomKFgenerator.true_kf(1)[0]
            numberSquares = 0
            for s in kf:
                if s.shape == "square":
                    numberSquares = numberSquares + 1
            if numberSquares > 1:
                kfs.append(kf)
                i = i + 1
        return kfs

    def false_kf(self, n=1):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 3, 3)
        while i < n:
            kf = randomKFgenerator.true_kf(1)[0]
            numberSquares = 0
            for s in kf:
                if s.shape == "square":
                    numberSquares = numberSquares + 1
            if numberSquares < 2:
                kfs.append(kf)
                i = i + 1
        return kfs

class Circles(KandinskyTruthInterfce):

    def isfuzzy(self):
        return false

    def humanDescription(self):
        return "contains a number of circles plus one random shape(square, triangle or circle)"

    def true_kf(self, expectedNrCircles=1, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, expectedNrCircles, expectedNrCircles + 3)
        while i < numberFigures:
            kf = randomKFgenerator.true_kf(1)[0]
            numberCircles = 0
            for s in kf:
                if s.shape == "circle":
                    numberCircles = numberCircles + 1
            if numberCircles == expectedNrCircles:
                kfs.append(kf)
                i = i + 1
        return kfs

    def false_kf(self, expectedNrCircles=1, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, expectedNrCircles, expectedNrCircles + 3)
        while i < numberFigures:
            kf = randomKFgenerator.true_kf(1)[0]
            numberCircles = 0
            for s in kf:
                if s.shape == "circle":
                    numberCircles = numberCircles + 1
            if numberCircles != expectedNrCircles:
                kfs.append(kf)
                i = i + 1
        return kfs
