import PIL
import random

from .KandinskyTruth import KandinskyTruthInterfce
from .KandinskyUniverse import kandinskyShape, overlaps


class Random(KandinskyTruthInterfce):

    def humanDescription(self):
        return "random kandinsky figure"

    def _randomobject(self, minsize=0.1, maxsize=0.5):
        o = kandinskyShape()
        o.color = random.choice(self.u.kandinsky_colors)
        o.shape = random.choice(self.u.kandinsky_shapes)
        o.size = minsize + (maxsize - minsize) * random.random()
        o.x = o.size / 2 + random.random() * (1 - o.size)
        o.y = o.size / 2 + random.random() * (1 - o.size)
        return o

    def _randomkf(self, min, max):
        kf = []
        kftemp = []
        n = random.randint(min, max)

        minsize = 0.1
        if n == 3: minsize = 0.2
        if n == 2: minsize = 0.3
        if n == 1: minsize = 0.4

        maxsize = 0.6
        if n == 5: maxsize = 0.5
        if n == 6: maxsize = 0.4
        if n == 7: maxsize = 0.3
        if n > 7: maxsize = 0.2

        i = 0
        maxtry = 20
        while i < n:
            kftemp = kf
            t = 0
            o = self._randomobject(minsize, maxsize)
            kftemp = kf[:]
            kftemp.append(o)
            while overlaps(kftemp) and (t < maxtry):
                o = self._randomobject(minsize, maxsize)
                kftemp = kf[:]
                kftemp.append(o)
                t = t + 1
            if (t < maxtry):
                kf = kftemp[:]
                i = i + 1
            else:
                maxsize = maxsize * 0.95
                # print(maxsize)
        return kf

    def _randomCircle(self, minsize=0.1, maxsize=0.5, color=None):
        o = kandinskyShape()
        if color:
            o.color = color
        else:
            o.color = random.choice(self.u.kandinsky_colors)

        o.shape = 'circle'
        o.size = minsize + (maxsize - minsize) * random.random()
        o.x = o.size / 2 + random.random() * (1 - o.size)
        o.y = o.size / 2 + random.random() * (1 - o.size)
        return o

    def circleskf(self):
        kf = []
        kftemp = []
        n = random.randint(self.min, self.max)

        minsize = 0.1
        if n == 3: minsize = 0.2
        if n == 2: minsize = 0.3
        if n == 1: minsize = 0.4

        maxsize = 0.6
        if n == 5: maxsize = 0.5
        if n == 6: maxsize = 0.4
        if n == 7: maxsize = 0.3
        if n > 7: maxsize = 0.2

        i = 0
        maxtry = 20
        while i < n:
            kftemp = kf
            t = 0
            o = self._randomCircle(minsize, maxsize)
            kftemp = kf[:]
            kftemp.append(o)
            while overlaps(kftemp) and (t < maxtry):
                o = self._randomCircle(minsize, maxsize)
                kftemp = kf[:]
                kftemp.append(o)
                t = t + 1
            if (t < maxtry):
                kf = kftemp[:]
                i = i + 1
            else:
                maxsize = maxsize * 0.95
                # print(maxsize)
        return kf

    def moreThanCircleskf(self, more, less):
        kf = []
        n = more + less

        moreColor = 'red'
        lessColor = 'blue'

        minsize = 0.1
        if n == 3: minsize = 0.2
        if n == 2: minsize = 0.3
        if n == 1: minsize = 0.4

        maxsize = 0.6
        if n == 5: maxsize = 0.5
        if n == 6: maxsize = 0.4
        if n == 7: maxsize = 0.3
        if n > 7: maxsize = 0.2

        # circles of the "more" class(red)
        i = 0
        maxtry = 20
        while i < more:
            t = 0
            o = self._randomCircle(minsize, maxsize, moreColor)
            kftemp = kf[:]
            kftemp.append(o)
            while overlaps(kftemp) and (t < maxtry):
                o = self._randomCircle(minsize, maxsize, moreColor)
                kftemp = kf[:]
                kftemp.append(o)
                t = t + 1
            if (t < maxtry):
                kf = kftemp[:]
                i = i + 1
            else:
                maxsize = maxsize * 0.95
                # print(maxsize)

        # circles of the "less" class(blue)
        i = 0
        maxtry = 20
        while i < less:
            t = 0
            o = self._randomCircle(minsize, maxsize, lessColor)
            kftemp = kf[:]
            kftemp.append(o)
            while overlaps(kftemp) and (t < maxtry):
                o = self._randomCircle(minsize, maxsize, lessColor)
                kftemp = kf[:]
                kftemp.append(o)
                t = t + 1
            if (t < maxtry):
                kf = kftemp[:]
                i = i + 1
            else:
                maxsize = maxsize * 0.95
                # print(maxsize)

        return kf

    def true_kf(self, n=1):
        kfs = []
        for i in range(n):
            kf = self._randomkf(self.min, self.max)
            kfs.append(kf)
        return kfs

    def ArithCirclesRpBeY(self, preDefinedCombos = []):
        kf = []
        if len(preDefinedCombos) > 0:
            index = random.randint(0, len(preDefinedCombos) - 1)
            number_red = preDefinedCombos[index][0]
            number_blue = preDefinedCombos[index][1]
        else:
            number_red = random.randint(1, 5)
            number_blue = random.randint(1, 5)

        number_yellow = number_red + number_blue

        shape_counts = [number_red, number_blue, number_yellow]

        n = number_red + number_blue + number_yellow

        minsize = 0.1
        if n == 3: minsize = 0.2
        if n == 2: minsize = 0.3
        if n == 1: minsize = 0.4

        maxsize = 0.6
        if n == 5: maxsize = 0.5
        if n == 6: maxsize = 0.4
        if n == 7: maxsize = 0.3
        if n > 7: maxsize = 0.2


        color_index = 0
        for color in ['red', 'blue', 'yellow']:
            i = 0
            maxtry = 20
            while i < shape_counts[color_index]:
                t = 0
                o = self._randomCircle(minsize, maxsize, color)
                kftemp = kf[:]
                kftemp.append(o)
                while overlaps(kftemp) and (t < maxtry):
                    o = self._randomCircle(minsize, maxsize, color)
                    kftemp = kf[:]
                    kftemp.append(o)
                    t = t + 1
                if (t < maxtry):
                    kf = kftemp[:]
                    i = i + 1
                else:
                    maxsize = maxsize * 0.95
            color_index = color_index + 1

        return kf

    def ArithCirclesRpBNoteY(self, red=None, blue=None):
        kf = []
        if red != None and blue != None:
            number_red = red
            number_blue = blue
        else:
            number_red = random.randint(1, 3)
            number_blue = random.randint(1, 3)


        number_yellow = random.randint(1, 9)

        while number_yellow == number_blue + number_red:
            number_yellow = random.randint(1, 9)

        shape_counts = [number_red, number_blue, number_yellow]

        n = number_red + number_blue + number_yellow

        minsize = 0.1
        if n == 3: minsize = 0.2
        if n == 2: minsize = 0.3
        if n == 1: minsize = 0.4

        maxsize = 0.6
        if n == 5: maxsize = 0.5
        if n == 6: maxsize = 0.4
        if n == 7: maxsize = 0.3
        if n > 7: maxsize = 0.2


        color_index = 0
        for color in ['red', 'blue', 'yellow']:
            i = 0
            maxtry = 20
            while i < shape_counts[color_index]:
                t = 0
                o = self._randomCircle(minsize, maxsize, color)
                kftemp = kf[:]
                kftemp.append(o)
                while overlaps(kftemp) and (t < maxtry):
                    o = self._randomCircle(minsize, maxsize, color)
                    kftemp = kf[:]
                    kftemp.append(o)
                    t = t + 1
                if (t < maxtry):
                    kf = kftemp[:]
                    i = i + 1
                else:
                    maxsize = maxsize * 0.95
            color_index = color_index + 1

        return kf