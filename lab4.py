# Optimizacijos Metodai Lab4
# Tomas Kvaraciejus, 2110588

import math
import numpy as np

matrixA =  [[-1,  1, -1, -1,  8],
            [ 2,  4,  0,  0, 10],
            [ 0,  0,  1,  1,  3],
            [ 2, -3,  0, -5,  0]]
matrixB =  [[-1,  1, -1, -1,  5],
            [ 2,  4,  0,  0,  8],
            [ 0,  0,  1,  1,  8],
            [ 2, -3,  0, -5,  0]]

# get most negative number in bottom row, that identified column
# find smallest quotient
# equal out member to 1, all other rows to 0, repeat

def createTable(inputMatrix):
    matrix = np.array(inputMatrix)
    cols, rows = matrix.shape[0], matrix.shape[1]
    identityMatrix = np.eye(cols)
    return np.hstack([matrix[:, :rows-1], identityMatrix, matrix[:, rows-1:rows]])

def getQuotient(inputMatrix, row, col):
    if inputMatrix[row][col] <= 0:
        return math.inf
    
    return inputMatrix[row][-1] / inputMatrix[row][col]

def adjustMatrix(inputMatrix, tRow, tCol):
    pivotValue = inputMatrix[tRow][tCol]
    inputMatrix[tRow] = [value / pivotValue for value in inputMatrix[tRow]]

    for row in range(len(inputMatrix)):
        if row == tRow:
            continue

        pivotValue = inputMatrix[row][tCol]
        inputMatrix[row] = [value - pivotValue * inputMatrix[tRow][col] for col,value in enumerate(inputMatrix[row])]

    return inputMatrix

def optimize(inputMatrix):
    inputMatrix = createTable(inputMatrix)
    rows, cols = len(inputMatrix), len(inputMatrix[0])

    while all(value >= 0 for value in inputMatrix[-1][:-1]) == False:
        _, tCol = min((inputMatrix[-1][tCol], tCol) for tCol in range(cols - 1))
        _, tRow = min((getQuotient(inputMatrix, tRow, tCol), tRow) for tRow in range(rows - 1))

        inputMatrix = adjustMatrix(inputMatrix, tRow, tCol)

    return inputMatrix

def printMatrixResult(iMatrix, oMatrix):
    rows, cols = len(iMatrix), len(iMatrix[0])
    varNames = [f"x{n + 1}" for n in range(rows)] + [f"a{n + 1}" for n in range(cols - 1)] + ['z', 'b']

    format_string = "{:<7}" * oMatrix.shape[1]
    print((format_string).format(*varNames))
    for row in oMatrix:
        print((format_string).format(*row))

    varValues = []
    for col in range(cols - 1):
        index = np.where(oMatrix[:,col])[0]
        if len(index) == 1 and all(oMatrix[row][col] == 0 or row == index[0] for row in range(rows)):
            value = oMatrix[index[0]][-1]
        else:
            value = 0
            
        varValues.append(f"x{col + 1} = {value}")

    print()
    print(varValues)

class lab4:
    inputMatrix = matrixB

    optimizedMatrix = (optimize(inputMatrix))
    printMatrixResult(inputMatrix, optimizedMatrix)

    