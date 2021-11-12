from classifier import Classifier
import pandas as pd
import numpy as np





class KNN(Classifier):

    def __init__(self, csv_file, debugging, k):
        self._k = k
        Classifier.__init__(self, csv_file, debugging)

    def _distance(self, rec1, rec2):
        np_array_r1 = np.asarray(rec1, dtype=np.float32)
        np_array_r2 = np.asarray(rec2, dtype=np.float32)
        return np.linalg.norm(np_array_r1, np_array_r2)

    def _count(self, L):
        C = {}

        for t in L:
            Y = t[0]
            if C.__contains__(Y):
                C[Y] = C[Y] + 1
            else:
                C[Y] = 1

        return C




    def classify_record(self, record):
        '''
        Classifica um registro
        '''
        data = pd.DataFrame(self._data).select_dtypes(['number']).to_numpy() #remove atributos qualitativos

        distances_list = []

        aux_colname = data.columns.values

        r2 = []

        if (data.columns.size > 0):
            for i in range(data[aux_colname[0]].size):
                for j in range(data.columns.size):
                    r2.append(data[aux_colname[j]][i])

                if (record != r2): #verifica se o registro não é ele mesmo
                    distances_list.append([self._data[-1][i], self._distance(record, r2)])

        distances_list.sort(key=lambda tup:tup[1])

        distances_list.remove(distances_list[k:-1])

        self._count(distances_list, self._k)









