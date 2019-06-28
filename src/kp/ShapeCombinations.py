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
            kf = randomKFgenerator.true_kf(1)
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


class CirclesOnly(KandinskyTruthInterfce):

    def isfuzzy(self):
        return false

    def humanDescription(self):
        return "contains a number of circles, no other forms)"

    def true_kf(self, expectedNrCircles=1, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 1, expectedNrCircles + 3)
        while i < numberFigures:
            kf = randomKFgenerator.circleskf()
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
            kf = randomKFgenerator.circleskf()
            numberCircles = 0
            for s in kf:
                if s.shape == "circle":
                    numberCircles = numberCircles + 1
            if numberCircles != expectedNrCircles:
                kfs.append(kf)
                i = i + 1
        return kfs


class CirclesOnlyRange(KandinskyTruthInterfce):

    def isfuzzy(self):
        return false

    def humanDescription(self):
        return "contains a number of circles, ranging from min to max"

    def true_kf(self, minExpectedNrCircles=1, maxExpectedCircles=2, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, minExpectedNrCircles, maxExpectedCircles)
        while i < numberFigures:
            kf = randomKFgenerator.circleskf()
            numberCircles = 0
            for s in kf:
                if s.shape == "circle":
                    numberCircles = numberCircles + 1
            if numberCircles >= minExpectedNrCircles and numberCircles <= maxExpectedCircles:
                kfs.append(kf)
                i = i + 1
        return kfs

    def false_kf(self, minExpectedNrCircles=1, maxExpectedCircles=2, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 0, maxExpectedCircles + 2)
        while i < numberFigures:
            kf = randomKFgenerator.circleskf()
            numberCircles = 0
            for s in kf:
                if s.shape == "circle":
                    numberCircles = numberCircles + 1
            if numberCircles < minExpectedNrCircles or numberCircles > maxExpectedCircles:
                kfs.append(kf)
                i = i + 1
        return kfs


class SetMoreRedThanBlue(KandinskyTruthInterfce):

    def isfuzzy(self):
        return false

    def humanDescription(self):
        return "contains more red circles than blue ones"

    def true_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.circleskf()
            numberRedCircles = 0
            numberBlueCircles = 0
            for s in kf:
                if s.color == "red":
                    numberRedCircles = numberRedCircles + 1
                else:
                    numberBlueCircles = numberBlueCircles + 1
            if numberRedCircles > numberBlueCircles and numberBlueCircles > 0:
                kfs.append(kf)
                i = i + 1
                if i % 100 == 0:
                    print(str(i) + '/' + str(numberFigures))
        return kfs

    def false_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.circleskf()
            numberRedCircles = 0
            numberBlueCircles = 0
            for s in kf:
                if s.color == "red":
                    numberRedCircles = numberRedCircles + 1
                else:
                    numberBlueCircles = numberBlueCircles + 1
            if numberRedCircles <= numberBlueCircles and numberRedCircles > 0:
                kfs.append(kf)
                i = i + 1
                if i % 100 == 0:
                    print(str(i) + '/' + str(numberFigures))
        return kfs


class SetMoreRedThanBlueSparse(KandinskyTruthInterfce):

    def isfuzzy(self):
        return false

    def humanDescription(self):
        return "contains more red circles than blue ones, training != test"

    def train_true_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.circleskf()
            numberRedCircles = 0
            numberBlueCircles = 0
            for s in kf:
                if s.color == "red":
                    numberRedCircles = numberRedCircles + 1
                else:
                    numberBlueCircles = numberBlueCircles + 1
            if numberRedCircles > numberBlueCircles and numberBlueCircles > 0 and not (
                    numberRedCircles == 4 and numberBlueCircles == 2):
                kfs.append(kf)
                i = i + 1
                if i % 100 == 0:
                    print(str(i) + '/' + str(numberFigures))
        return kfs

    def train_false_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.circleskf()
            numberRedCircles = 0
            numberBlueCircles = 0
            for s in kf:
                if s.color == "red":
                    numberRedCircles = numberRedCircles + 1
                else:
                    numberBlueCircles = numberBlueCircles + 1
            if numberRedCircles <= numberBlueCircles and numberBlueCircles > 0 and not (
                    numberRedCircles == 2 and numberBlueCircles == 4):
                kfs.append(kf)
                i = i + 1
                if i % 2 == 0:
                    print(str(i) + '/' + str(numberFigures))
        return kfs

    def test_true_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.moreThanCircleskf(4, 2)
            kfs.append(kf)
            i = i + 1
            if i % 100 == 0:
                print(str(i) + '/' + str(numberFigures))
        return kfs

    def test_false_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.moreThanCircleskf(2, 4)
            kfs.append(kf)
            i = i + 1
            if i % 2 == 0:
                print(str(i) + '/' + str(numberFigures))
        return kfs


class ArithRplusBisY(KandinskyTruthInterfce):

    def isfuzzy(self):
        return false

    def humanDescription(self):
        return "number plus number blue equals number yellow, training != test"

    def train_true_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.ArithCirclesRpBeY()
            numberRedCircles = 0
            numberBlueCircles = 0
            numberYellowCircles = 0
            for s in kf:
                if s.color == "red":
                    numberRedCircles = numberRedCircles + 1
                elif s.color == 'blue':
                    numberBlueCircles = numberBlueCircles + 1
                else:
                    numberYellowCircles = numberYellowCircles + 1
            if ((numberRedCircles + numberBlueCircles) == numberYellowCircles) \
                    and not (numberRedCircles == 4 and numberBlueCircles == 2) \
                    and not (numberRedCircles == 1 and numberBlueCircles == 3) \
                    and not (numberRedCircles == 2 and numberBlueCircles == 5)\
                    and not (numberRedCircles == 4 and numberBlueCircles == 6):
                kfs.append(kf)
                i = i + 1
                if i % 100 == 0:
                    print(str(i) + '/' + str(numberFigures))
        return kfs

    def train_false_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.circleskf()
            numberRedCircles = 0
            numberBlueCircles = 0
            numberYellowCircles = 0
            for s in kf:
                if s.color == "red":
                    numberRedCircles = numberRedCircles + 1
                elif s.color == 'blue':
                    numberBlueCircles = numberBlueCircles + 1
                else:
                    numberYellowCircles = numberYellowCircles + 1
            if not ((numberRedCircles + numberBlueCircles) == numberYellowCircles) and not (
                    numberRedCircles == 3 and numberBlueCircles == 2) and not (
                    numberRedCircles == 1 and numberBlueCircles == 3):
                kfs.append(kf)
                i = i + 1
                if i % 2 == 0:
                    print(str(i) + '/' + str(numberFigures))
        return kfs

    def test_true_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.ArithCirclesRpBeY([[4, 2], [1, 3], [2, 5], [4, 6]])
            kfs.append(kf)
            i = i + 1
            if i % 100 == 0:
                print(str(i) + '/' + str(numberFigures))
        return kfs

    def test_false_kf(self, numberFigures=50):
        kfs = []
        i = 0
        randomKFgenerator = Random(self.u, 2, 10)
        while i < numberFigures:
            kf = randomKFgenerator.ArithCirclesRpBNoteY(3, 2)
            numberRedCircles = 0
            numberBlueCircles = 0
            numberYellowCircles = 0
            for s in kf:
                if s.color == "red":
                    numberRedCircles = numberRedCircles + 1
                elif s.color == 'blue':
                    numberBlueCircles = numberBlueCircles + 1
                else:
                    numberYellowCircles = numberYellowCircles + 1
            if not ((numberRedCircles + numberBlueCircles) == numberYellowCircles)and (
                    (numberRedCircles == 3 and numberBlueCircles == 2) or (
                    numberRedCircles == 1 and numberBlueCircles == 3)):
                kfs.append(kf)
                i = i + 1
                if i % 2 == 0:
                    print(str(i) + '/' + str(numberFigures))
        return kfs
