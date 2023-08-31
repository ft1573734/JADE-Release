from itertools import count
import cupy as cp
import numpy as np

class EmbeddedColumn:
    def __init__(self):
        self.Index = -1
        self.OriginalColumn = []
        self.V2Matrix = []
        self.APLs = []
        self.length = -1

        
class JADECore:
    GLOBAL_COUNTER = 0
    k = -1
    def __init__(self, EmbeddingModel):
        self.embed = EmbeddingModel

    def GenerateV2Matrix(self, column):
        V2Matrix = cp.ndarray(shape = (len(column),self.embed.Dimensionality), dtype=float, order='C')
        for i in range(0, len(column)):
            V2Matrix[i] = cp.array(self.embed.Embed(inputText = column[i]))
        return V2Matrix

    def CalculateAPL(self, V2Matrix):
        BasicVectors = self.embed.BasicVectors
        APLs = cp.sum(cp.matmul(V2Matrix,BasicVectors), axis = 0)/len(V2Matrix)
        return APLs
    
    def CalculateAPLDist(self, col1, col2):
        return max(cp.abs(col1.APLs - col2.APLs))

    def CalculateJaccardSimilarityUsingGEMM(self, V2Matrix_1, V2Matrix_2):
        resultMatrix = cp.matmul(V2Matrix_1,cp.transpose(V2Matrix_2))
        count_ones = len(cp.argwhere(resultMatrix > 0.9999999)) #We cannot use '==1' due to precision losses during execution
        dist = count_ones/(len(V2Matrix_1) + len(V2Matrix_2) - count_ones)
        return dist



    #Find the joinable columns of a given column (targetColumn) within a data set (dataset). 
    #Parameters: 
    #   dataset: A list of columns. Each column is stored using the 'EmbeddedColumn' structure
    #   targetColumn: A given column stored in the 'EmbeddedColumn' structure
    #   threshold: A variable between [0,1], columns whose Jaccard similarity are above this threshold are considered as joinable
    #Output:
    #   result: Indices of the joinable columns of the targetColumn, the index refers to the index within the data set (dataset).
    def SearchJoinableColumn_Threshold(self, dataset, targetColumn, threshold):
        result = []
        dist_threshold = 1 - threshold
        for i in range(0,len(dataset)):
            #First, filtering the joinable columns using APL
            column = dataset[i]
            if self.CalculateAPLDist(column, targetColumn) > dist_threshold:
                #print("Successfully skipped")
                continue #If the APL dist is greater than the threshold dist, then col1 and col2 are guaranteed to be not joinable
            else:
                #Else, we need to evaluate their similarity in a more accurate way.
                #print("Not skipped")
                tmpSim = self.CalculateJaccardSimilarityUsingGEMM(column.V2Matrix, targetColumn.V2Matrix)
                if tmpSim >= dist_threshold:
                    result.append(i)
        return result


    def GenerateAPLMatrix(self, EmbeddedColumns, BasicVectorCollection):
        APLMatrix = cp.ndarray(shape = (len(EmbeddedColumns),len(BasicVectorCollection)), dtype=float, order='C')
        count_basicVectors = len(BasicVectorCollection)
        for i in range(0, len(EmbeddedColumns)):
            print(f"i = {i}, total: {len(EmbeddedColumns)}")
            for j in range(0, count_basicVectors):
                tmpColumn = EmbeddedColumns[i].V2Matrix
                tmpBasicVectorArray = BasicVectorCollection[j]
                tmpBasicVector = cp.asarray(tmpBasicVectorArray)
                PLs = cp.matmul(tmpColumn,cp.transpose(tmpBasicVector))
                tmpAPL = sum(PLs)/count_basicVectors
                APLMatrix[i][j] = tmpAPL
        return APLMatrix

    def ConcatenateDataset(self, Columns):
        maps = [0]
        ColumnIndices = [-1]
        V2Matrix_array = []
        index = 0
        for column in Columns:
            ColumnIndices.append(column.Index)
            V2Matrix_array.append(column.V2Matrix)
            index += column.length
            maps.append(index)
        return cp.concatenate(V2Matrix_array, axis = 0), maps, ColumnIndices

    def CalculateJaccardSimilarityUsingConcatenatedV2Matrices(self, GivenColumn, ConcatenatedV2Matrices, Maps, ColumnIndices):
        resultMatrix = cp.matmul(ConcatenatedV2Matrices, cp.transpose(GivenColumn.V2Matrix))
        Sims = []
        for i in range(0, len(Maps)-1):
            tmpResult = resultMatrix[Maps[i]:Maps[i+1],:]
            count_ones = len(cp.argwhere(tmpResult > 0.9999999))
            if GivenColumn.length + len(tmpResult) <= 2 * count_ones:
                Sims.append(1)
            else:
                dist = count_ones/(GivenColumn.length + len(tmpResult) - count_ones)
                Sims.append(dist)
        return Sims

    def CalculateDistMatrixViaAPLMatrix(self, DSInAPLMatrixForm, TargetColumnsInAPLForm, threshold):
        count_TargetColumns = TargetColumnsInAPLForm.shape[0]
        count_DSColumns = DSInAPLMatrixForm.shape[0]
        result = cp.ndarray(shape = (count_TargetColumns, count_DSColumns), dtype = float, order = 'C')
        for i in range(0,count_TargetColumns):
            for j in range(0, count_DSColumns):
                result[i][j] = cp.abs(TargetColumnsInAPLForm[i]-DSInAPLMatrixForm[j]).max()
        return result

    def FilterViaAPLMatrix(TargetColumnAPL, APLMatrix, DistThreshold):
        dim = APLMatrix.shape # n represents # columns in the data set

        tmpAPLColumn = cp.ndarray(shape = (1, dim[1]),dtype = float)
        tmpAPLColumn[0,:] = cp.array(TargetColumnAPL)

        TargetColumnDuplicatedMatrix = cp.repeat(tmpAPLColumn, dim[0], axis = 0)
        R_tmp = cp.where(cp.absolute(TargetColumnDuplicatedMatrix - APLMatrix) <= DistThreshold, x=1, y=0) # Checking which columns satisfies the APL theorem on each APL dimension
        return cp.where(cp.sum(R_tmp, axis = 1)==dim[1], x=1, y=0)