from importdata import importasline
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

'''
	This code set implement grouping schemes
	Each grouping schemes returns data sets for training

	Note:
			filename can only take '../data/shakespear_modified.txt' 
								   '../data/spenser_modified.txt'
								   '../data/all_modified.txt'
'''

def grouping1(filename):

	###
    ###	scheme	ABAB CDCD EFEF GG
    ###

    dataSet = importasline(filename)

    assert 0 == len(dataSet) % 14, 'number of lines should be multiple of 14' 
    num_poem = len(dataSet) / 14
    groupA = [] ; groupB = [] ; groupC = [] ; groupD = [] ; groupE = [] ; groupF = [] ; groupG = []

    for it in range(num_poem):
    	start = it * 14
    	groupA.append(dataSet[start + 0]) ; groupA.append(dataSet[start + 2])
    	groupB.append(dataSet[start + 1]) ; groupB.append(dataSet[start + 3])
    	groupC.append(dataSet[start + 4]) ; groupC.append(dataSet[start + 6])
    	groupD.append(dataSet[start + 5]) ; groupD.append(dataSet[start + 7])
    	groupE.append(dataSet[start + 8]) ; groupE.append(dataSet[start +10])
    	groupF.append(dataSet[start + 9]) ; groupF.append(dataSet[start +11])
    	groupG.append(dataSet[start +12]) ; groupG.append(dataSet[start +13])

    assert groupA[1] == dataSet[2], 'groupA[1] == dataSet[2] should hold'
    assert groupC[1] == dataSet[6], 'groupC[1] == dataSet[6] should hold'

    return groupA, groupB, groupC, groupD, groupE, groupF, groupG

def grouping2(filename):

	###
    ###	scheme	AAAA BBBB CCCC DD
    ###

    dataSet = importasline(filename)

    assert 0 == len(dataSet) % 14, 'number of lines should be multiple of 14' 
    num_poem = len(dataSet) / 14
    groupA = [] ; groupB = [] ; groupC = [] ; groupD = []

    for it in range(num_poem):
    	start = it * 14
    	groupA.append(dataSet[start + 0]) ; groupA.append(dataSet[start + 1]) ; groupA.append(dataSet[start + 2]) ; groupA.append(dataSet[start + 3])
    	groupB.append(dataSet[start + 4]) ; groupB.append(dataSet[start + 5]) ; groupB.append(dataSet[start + 6]) ; groupB.append(dataSet[start + 7]) 
    	groupC.append(dataSet[start + 8]) ; groupC.append(dataSet[start + 9]) ; groupC.append(dataSet[start +10]) ; groupC.append(dataSet[start +11])
    	groupD.append(dataSet[start +12]) ; groupD.append(dataSet[start +13])


    assert groupB[3] == dataSet[7], 'groupB[3] == dataSet[7] should hold'
    assert groupC[1] == dataSet[9], 'groupC[1] == dataSet[9] should hold'

    return groupA, groupB, groupC, groupD

def grouping3(filename):

	###
    ###	scheme	AAAA AAAA AAAA BB
    ###

    dataSet = importasline(filename)

    assert 0 == len(dataSet) % 14, 'number of lines should be multiple of 14' 
    num_poem = len(dataSet) / 14
    groupA = [] ; groupB = [] ; groupC = [] ; groupD = []

    for it in range(num_poem):
    	start = it * 14
    	groupA.append(dataSet[start + 0]) ; groupA.append(dataSet[start + 1]) ; groupA.append(dataSet[start + 2]) ; groupA.append(dataSet[start + 3]) ;\
    	groupA.append(dataSet[start + 4]) ; groupA.append(dataSet[start + 5]) ; groupA.append(dataSet[start + 6]) ; groupA.append(dataSet[start + 7]) ;\
    	groupA.append(dataSet[start + 8]) ; groupA.append(dataSet[start + 9]) ; groupA.append(dataSet[start +10]) ; groupA.append(dataSet[start +11]) ;\
    	groupB.append(dataSet[start +12]) ; groupB.append(dataSet[start +13])

    assert groupB[1] == dataSet[13], 'groupB[1] == dataSet[13] should hold'

    return groupA, groupB


def main():
    [groupA,groupB,groupC,groupD,groupE,groupF,groupG]=grouping1('../data/shakespear_modified.txt')
    np.savetxt('../data/grouping1/groupA.txt',groupA,fmt='%s')
    np.savetxt('../data/grouping1/groupB.txt',groupB,fmt='%s')
    np.savetxt('../data/grouping1/groupC.txt',groupC,fmt='%s')
    np.savetxt('../data/grouping1/groupD.txt',groupD,fmt='%s')
    np.savetxt('../data/grouping1/groupE.txt',groupE,fmt='%s')
    np.savetxt('../data/grouping1/groupF.txt',groupF,fmt='%s')
    np.savetxt('../data/grouping1/groupG.txt',groupG,fmt='%s')

if __name__ == "__main__":
    main()

