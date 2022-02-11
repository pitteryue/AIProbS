import math
import numpy as np

class DHC_extension():

    def UpdataDistribution(self, neighborhood, current_distribution):  #
        # neighborhood is a nested list:[[],[],...,[]] containing every node's neighborhood information
        updated_distribution = np.zeros(len(neighborhood))
        for i in range(len(neighborhood)):
            neih_Hi = np.sort(current_distribution[neighborhood[i]])[::-1]
            Hi = np.sum(neih_Hi - np.arange(1,len(neih_Hi)+1) >= 0)
            updated_distribution[i] = Hi

        return updated_distribution

    def ShannonEntropy(self, H):
        shannon_entropy = 0
        total = len(H)
        Hi_Counts = {}
        for i in range(len(H)):
            Hi = H[i]
            if Hi_Counts.get(Hi) is None:
                Hi_Counts[Hi] = 1
            else:
                Hi_Counts[Hi] += 1

        for key in Hi_Counts.keys():
            prob = float(Hi_Counts[key])/ total
            shannon_entropy -= prob * math.log(prob, 2)

        return shannon_entropy