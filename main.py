import math, sys, getopt, random, operator, pandas as pd, numpy as np, os

# Parameters
displayWidth = 800
displayHeight = 480

# wArea = 12
wArea = 18
wBalance = 0.5
wEquilibrium = 1
wSymmetric = 1
wSequence = 0.5
wConcentric = 1


class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def getArea(self):
        area = (self.x2 - self.x1) * (self.y2 - self.y1)
        if area == 0:
            # print("area is 0, (", self.x1, ", ", self.y1, "), (", self.x2, ", ", self.y2, ")")
            return 1
        return area

    def getPoint(self, i):
        if i == 0:
            return self.x1
        elif i == 1:
            return self.y1
        elif i == 2:
            return self.x2
        else:
            return self.y2

    def setPoint(self, i, delta):
        if i == 0:
            self.x1 += delta
        elif i == 1:
            self.y1 += delta
        elif i == 2:
            self.x2 += delta
        else:
            self.y2 += delta

    def meet(self, rect):
        possibleX = False
        possibleY = False

        if rect.x1 < self.x1 < rect.x2 or rect.x1 < self.x2 < rect.x2 or self.x1 < rect.x1 < self.x2 or self.x1 < rect.x2 < self.x2:
            possibleX = True
        else:
            return False

        if rect.y1 < self.y1 < rect.y2 or rect.y1 < self.y2 < rect.y2 or self.y1 < rect.y1 < self.y2 or self.y1 < rect.y2 < self.y2:
            possibleY = True
        else:
            return False

        if rect.x1 < self.x1 < rect.x2 and rect.x1 < self.x2 < rect.x2:
            if rect.y1 < self.y1 < rect.y2 and rect.y1 < self.y2 < rect.y2:
                return False

        if self.x1 < rect.x1 < self.x2 and self.x1 < rect.x2 < self.x2:
            if self.y1 < rect.y1 < self.y2 and self.y1 < rect.y2 < self.y2:
                return False

        return possibleX and possibleY

    def __repr__(self):
        return "(" + str(self.x1) + ", " + str(self.y1) + ")" + " ~ (" + str(self.x2) + ", " + str(self.y2) + ")"


class Layout:
    def __init__(self, layout):
        self.layout = layout

    def getArea(self):
        area = 0

        for i in range(0, len(self.layout)):
            area += self.layout[i].getArea()

        if area == 0:
            return 1

        return area

    def getAverageArea(self):
        area = 0

        for i in range(0, len(self.layout)):
            area += self.layout[i].getArea()

        area /= len(self.layout)

        if area == 0:
            print("Area is zero")
            return 1

        return area

    def getBalanceX(self):
        balanceX = 0

        for i in range(0, len(self.layout)):
            middle = (self.layout[i].x2 - self.layout[i].x1) / 2 + self.layout[i].x1
            balanceX += (displayWidth / 2 - middle) * self.layout[i].getArea()

        balanceX = abs(balanceX)
        balanceX /= len(self.layout)
        balanceX /= (displayWidth * displayHeight / 2) * (displayWidth / 4)

        if balanceX > 1:
            print("BalanceX is too high: ", str(balanceX))
            return 1

        return balanceX

    def getBalanceY(self):
        balanceY = 0

        for i in range(0, len(self.layout)):
            middle = (self.layout[i].y2 - self.layout[i].y1) / 2 + self.layout[i].y1
            balanceY += (displayHeight / 2 - middle) * self.layout[i].getArea()

        balanceY = abs(balanceY)
        balanceY /= len(self.layout)
        # balanceY /= (displayWidth * displayHeight / 2) * (displayWidth / 4)
        balanceY /= (displayWidth * displayHeight / 2) * (displayWidth / 3)

        if balanceY > 1:
            print("BalanceY is too high: ", str(balanceY))
            return 1

        return balanceY

    def getEquilibrium(self):
        equilibrium = 0
        equilibriumX = 0
        equilibriumY = 0

        for i in range(0, len(self.layout)):
            equilibriumX += (displayWidth / 2 - ((self.layout[i].x2 - self.layout[i].x1) / 2 + self.layout[i].x1)) * self.layout[i].getArea()
            equilibriumY += (displayHeight / 2 - ((self.layout[i].y2 - self.layout[i].y1) / 2 + self.layout[i].y1)) * self.layout[i].getArea()

        equilibriumX = abs(equilibriumX)
        equilibriumY = abs(equilibriumY)

        equilibriumX /= self.getArea()
        equilibriumY /= self.getArea()

        equilibrium = (equilibriumX / displayWidth + equilibriumY / displayHeight) / 2

        if equilibrium > 1:
            print("Equilibrium is too high: ", str(equilibrium))
            equilibrium = 1

        return equilibrium

    def getSymmetric(self):
        symmetric = 0
        horizontalSymmetric = 0
        verticalSymmetric = 0

        for i in range(0, len(self.layout)):
            horizontalSymmetric += (displayWidth / 2) - ((self.layout[i].x2 - self.layout[i].x1) / 2 + self.layout[i].x1)
            verticalSymmetric += (displayHeight / 2) - ((self.layout[i].y2 - self.layout[i].y1) / 2 + self.layout[i].y1)

        horizontalSymmetric = abs(horizontalSymmetric)
        verticalSymmetric = abs(verticalSymmetric)

        horizontalSymmetric /= (displayWidth / 2)
        verticalSymmetric /= (displayHeight / 2)

        symmetric = (horizontalSymmetric + verticalSymmetric) / 2

        return symmetric

    def getSequence(self):
        sequence = 0
        right = 0
        bottom = 0
        quad = 0

        for i in range(0, len(self.layout)):
            if ((self.layout[i].x2 - self.layout[i].x1) / 2 + self.layout[i].x1) - (displayWidth / 2) > 0:
                right = 1
            else:
                right = -1

            if ((self.layout[i].y2 - self.layout[i].y1) / 2 + self.layout[i].y1) - (displayHeight / 2) > 0:
                bottom = 1
            else:
                bottom = -1

            if right == -1 and bottom == -1:
                quad = 4
            elif right == 1 and bottom == -1:
                quad = 3
            elif right == -1 and bottom == 1:
                quad = 2
            else:
                quad = 1

            sequence += self.layout[i].getArea() * quad

        sequence /= (self.getArea() * 4)

        return sequence

    def getConcentric(self):
        concentric = 0

        diagonal = math.sqrt((displayWidth / 2) ** 2 + (displayHeight / 2) ** 2)

        for i in range(0, len(self.layout)):
            concentric += math.sqrt((((self.layout[i].x2 - self.layout[i].x1) / 2 + self.layout[i].x1) - (displayWidth / 2)) ** 2 + (((self.layout[i].y2 - self.layout[i].y1) / 2 + self.layout[i].y1) - (displayHeight / 2)) ** 2)

        concentric /= len(self.layout)
        concentric /= diagonal

        return concentric

    def getFitness(self):
        fitness = 0.0
        resolution = displayWidth * displayHeight
        diagonal = math.sqrt((displayWidth / 2) ** 2 + (displayHeight / 2) ** 2)

        area = (resolution - self.getAverageArea()) / resolution  # 작을수록 좋음
        if area < 0:
            area = 0

        fitness += area * wArea
        fitness += self.getBalanceX() * wBalance  # 작을수록 좋음
        fitness += self.getBalanceY() * wBalance  # 작을수록 좋음
        fitness += self.getEquilibrium() * wEquilibrium  # 작을수록 좋음
        fitness += self.getSymmetric() * wSymmetric  # 작을수록 좋음
        fitness += self.getSequence() * wSequence  # 작을수록 좋음
        fitness += self.getConcentric() * wConcentric  # 작을수록 좋음

        fitness /= wArea + wBalance + wEquilibrium + wSymmetric + wSequence + wConcentric

        return fitness

    def meetAll(self, target):
        for i in range(0, len(self.layout)):
            if i == target:
                continue

            if self.layout[target].meet(self.layout[i]):
                return True

        return False

    def getDirection(self, i, j):
        originFitness = self.getFitness()

        self.layout[i].setPoint(j, 1)
        nextFitness = self.getFitness()

        self.layout[i].setPoint(j, -2)
        prevFitness = self.getFitness()

        self.layout[i].setPoint(j, 1)

        if originFitness <= nextFitness and originFitness <= prevFitness:
            return 0
        elif nextFitness < originFitness and nextFitness < prevFitness:
            return 1
        else:
            return -1

    def getDelta(self, rectIdx, j, direction):
        originFitness = self.getFitness()
        fitnessArray = []

        for count in range(0, 10):
            fitnessArray.append(999999)
            delta = (2 ** count) * direction

            if j == 0 or j == 2:
                if self.layout[rectIdx].getPoint(j) + delta < 0 or self.layout[rectIdx].getPoint(j) + delta >= displayWidth:
                    continue

            if j == 1 or j == 3:
                if self.layout[rectIdx].getPoint(j) + delta < 0 or self.layout[rectIdx].getPoint(j) + delta >= displayHeight:
                    continue

            if j == 0 or j == 1:
                if self.layout[rectIdx].getPoint(j) + delta >= self.layout[rectIdx].getPoint(j + 2):
                    continue

            if j == 2 or j == 3:
                if self.layout[rectIdx].getPoint(j) + delta <= self.layout[rectIdx].getPoint(j - 2):
                    continue

            self.layout[rectIdx].setPoint(j, delta)

            # print("fitnessArray count:", count, ", value:", fitnessArray[count])

            if not self.meetAll(rectIdx):
                if self.layout[rectIdx].getArea() < displayWidth * displayHeight / 4:
                    fitnessArray[count] = self.getFitness()

            self.layout[rectIdx].setPoint(j, -delta)

        fitnessBest = 999999
        fitnessBestIndex = 0
        for idx in range(0, 10):
            if fitnessArray[idx] < fitnessBest:
                fitnessBestIndex = idx
                fitnessBest = fitnessArray[idx]

        if fitnessArray[fitnessBestIndex] > originFitness:
            return 0

        return (2 ** fitnessBestIndex) * direction

    def getRandomeDelta(self, rectIdx, j, direction):
        fitnessArray = []

        for count in range(0, 10):
            fitnessArray.append(999999)
            delta = (2 ** count) * direction

            if j == 0 or j == 2:
                if self.layout[rectIdx].getPoint(j) + delta < 0 or self.layout[rectIdx].getPoint(j) + delta >= displayWidth:
                    continue

            if j == 1 or j == 3:
                if self.layout[rectIdx].getPoint(j) + delta < 0 or self.layout[rectIdx].getPoint(j) + delta >= displayHeight:
                    continue

            if j == 0 or j == 1:
                if self.layout[rectIdx].getPoint(j) + delta >= self.layout[rectIdx].getPoint(j + 2):
                    continue

            if j == 2 or j == 3:
                if self.layout[rectIdx].getPoint(j) + delta <= self.layout[rectIdx].getPoint(j - 2):
                    continue

            self.layout[rectIdx].setPoint(j, delta)

            if not self.meetAll(rectIdx):
                fitnessArray[count] = 1

            self.layout[rectIdx].setPoint(j, -delta)

        fitnessBestIndex = -1
        for idx in range(0, 10):
            if fitnessArray[idx] == 1:
                fitnessBestIndex = idx
                break

        if fitnessBestIndex == -1:
            return 0

        return (2 ** fitnessBestIndex) * direction


def initialPopulation(popCount, rectCount):
    population = []

    for i in range(0, popCount):
        individual = []

        for j in range(0, rectCount):
            x = random.randrange(0, displayWidth - 1)
            y = random.randrange(0, displayHeight - 1)
            individual.append(Rect(x, y, x + 1, y + 1))

        population.append(Layout(individual))

    return population


# Population을 Ranking 순으로 정렬하기
def rankPop(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = population[i].getFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=False)


# Ranking 순으로 소팅된 Population에서 부모를 선별하기 → 부모의 index 값을 모으기
def selectPop(pop, eliteSize):
    selectResults = []
    df = pd.DataFrame(np.array(pop), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectResults.append(pop[i][0])

    for i in range(0, len(pop) - eliteSize):
        pick = random.randrange(0, 100)
        for j in range(0, len(pop)):  # 랜덤 결과에 의해 중복으로 들어가는 Individual 존재
            if pick <= df.iat[j, 3]:
                selectResults.append(pop[j][0])
                break

    return selectResults


# 부모의 index 값으로 individual 모으기
def popToPopulation(population, popSelected):
    pool = []
    for i in range(0, len(popSelected)):
        index = popSelected[i]
        pool.append(population[index])
    return pool


def avmIndividual(individual):
    for i in range(0, len(individual.layout)):
        for j in range(0, 4):
            direction = individual.getDirection(i, j)
            if direction == 0:
                continue
            delta = individual.getDelta(i, j, direction)

            if (j == 0 and individual.layout[i].getPoint(0) + delta == individual.layout[i].getPoint(2)) or \
               (j == 1 and individual.layout[i].getPoint(1) + delta == individual.layout[i].getPoint(3)) or \
               (j == 2 and individual.layout[i].getPoint(2) + delta == individual.layout[i].getPoint(0)) or \
               (j == 3 and individual.layout[i].getPoint(3) + delta == individual.layout[i].getPoint(1)):
                print("FIXME: Same thing (", individual.layout[i].x1, ", ", individual.layout[i].y1, ", ", individual.layout[i].x2, ", ", individual.layout[i].y2, ")", j, " is ", delta)
            else:
                individual.layout[i].setPoint(j, delta)
                if individual.meetAll(i):
                    print("FIXME: ERROR, it meets other rectangles:", i)


def avmPopulation(population):
    for i in range(0, len(population)):
        avmIndividual(population[i])


def mutateIndividual(individual, mutationCount):
    for i in range(0, mutationCount):
        rectIndex = random.randrange(0, len(individual.layout))

        for j in range(0, 4):
            direction = random.randrange(0, 3)
            if direction == 0:
                continue

            delta = individual.getRandomeDelta(rectIndex, j, direction)
            if delta == 0:
                continue

            individual.layout[rectIndex].setPoint(j, delta)


def mutatePopulation(population, mutationRate):
    for i in range(0, len(population)):
        mutateIndividual(population[i], round(mutationRate * len(population[i].layout)))


def nextGeneration(population, eliteSize, mutationRate):
    popRanked = rankPop(population)
    popSelected = selectPop(popRanked, eliteSize)
    population = popToPopulation(population, popSelected)
    avmPopulation(population)
    mutatePopulation(population, mutationRate)
    return population


def writeHTML(individual, fitness):
    name = 'candidate_layout_' + str(round(fitness, 4)) + '.html'
    file = open(name, 'w', encoding='UTF-8')
    file.write("<!DOCTYPE html><html><head><style>")
    file.write("div.absolute { position: absolute; \
                top: 0px; \
                left: 0px; \
                width: " + str(displayWidth) + "px; \
                height: " + str(displayHeight) + "px; \
                border: 1px solid #FF0000; }    ")

    for i in range(0, len(individual.layout)):
        style = "div.absolute" + str(i) \
                + " { position: absolute; top: " + str(individual.layout[i].getPoint(1)) \
                + "px; left: " + str(individual.layout[i].getPoint(0)) \
                + "px; width: " + str(individual.layout[i].getPoint(2) - individual.layout[i].getPoint(0)) \
                + "px; height: " + str(individual.layout[i].getPoint(3) - individual.layout[i].getPoint(1)) \
                + "px; border: 1px solid #FF00FF; }"
        file.write(style)

    file.write("</style></head><body>")
    file.write("<div class='absolute'></div")

    for i in range(0, len(individual.layout)):
        div = '<div class="absolute' + str(i) + '"></div>'
        file.write(div)

    file.write("</body></html>")
    file.close()


def geneticAlgorithm(rectCount, popSize, eliteSize, mutationRate, generations):
    population = initialPopulation(popSize, rectCount)
    print("Initial fitness: " + str(population[0].getFitness()))

    for i in range(0, generations):
        population = nextGeneration(population, eliteSize, mutationRate)
        print(str(i + 2) + "th trial's fitness: " + str(population[0].getFitness()))

    print("Final fitness: " + str(population[0].getFitness()))

    for i in range(0, len(population[0].layout)):
        print(population[0].layout[i])

    writeHTML(population[0], population[0].getFitness())


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "p:f:")
    except getopt.GetoptError:
        print('test.py -p <population> -f <fitness evaluation> inFile')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-p':
            population = arg
        elif opt == '-f':
            fitnessEvaluation = arg

    print('Population:', population, ', Fitness Evaluation:', fitnessEvaluation)

    geneticAlgorithm(rectCount=10, popSize=100, eliteSize=20, mutationRate=0.1, generations=10)


if __name__ == "__main__":
    main(sys.argv[1:])

