#Jeremy Zackon
#CS378
#hw3

import sys
import math
import collections

#Node class that has a parent and list of children
#feature is the label of an attribute that we split on at this node
#featIndex is the index corresponding to the row of data
#rows is the sublist of data pertinent to this node
#classification is set to a class when we make a leaf node
class Node:
	def __init__(self, parent, childList, feat, fIndex, rows):
		self.parent = parent
		self.childList = childList
		self.feature = feat
		self.featIndex = fIndex
		self.rows = rows
		self.classification = None

#Calculates the expected information (entropy) for the class variable
def calcExpectedInfo(attDict):
	info = 0.0
	for att in attDict:
		pCount = float(attDict[att])
		total = float(sum(x for x in attDict.values()))
		prob = pCount/total
		info -= (prob*(math.log(prob, 2)))

	return info

#Calculates the information needed to split on an attribute
def calcInfoNeeded(dataRows, attDict, attDictIndex):
	info = 0.0
	for att in attDict:
		pCount = float(attDict[att])
		total = float(sum(x for x in attDict.values()))
		subDict = {}
		for row in dataRows:
			if row[attDictIndex] is att:
				if row[0] in subDict:
					subDict[row[0]] += 1
				else:
					subDict[row[0]] = 1

		info += (pCount/total) * calcExpectedInfo(subDict)

	return info

#Calculates splitinfo to calculate gain ratio
def splitInfo(attDict):
	split = 0.0
	for att in attDict:
		pCount = float(attDict[att])
		total = float(sum(x for x in attDict.values()))
		prob = pCount/total
		split -= (prob*(math.log(prob, 2)))

	return split

#Determines if the node is a leaf node (all records belong to one class)
def needsLeafNode(node):
	tempDict = {}
	for row in node.rows:
		if row[0] in tempDict:
			tempDict[row[0]] += 1
		else:
			tempDict[row[0]] = 1
	keys = list(tempDict.keys())

	if(len(keys) == 1):
		return keys[0]
	else:
		return ''

#Build the decision tree starting from the root node n
def buildTree(n):

	numAtts = len(n.rows[0])
	dictList = []

	for x in range(0, numAtts):
		d = {}
		dictList.append(d)

	for row in n.rows:
		for x in range(0, numAtts):
			if(row[x] in dictList[x]):
				dictList[x][row[x]] += 1
			else:
				dictList[x][row[x]] = 1

	leaf = needsLeafNode(n)
	if leaf is not '':
		n.classification = leaf
		return

	maxGainRatio = 0.0
	maxGRIndex = 0
	for x in range(1, len(dictList)):
		gain = float(calcExpectedInfo(dictList[0]) - calcInfoNeeded(n.rows, dictList[x], x))
		split = float(splitInfo(dictList[x]))
		if split == 0.0:
			gainRatio = 0
		else:
			gainRatio = gain/split
		if gainRatio > maxGainRatio:
			maxGainRatio = gainRatio
			maxGRIndex = x

	for att in dictList[maxGRIndex]:
		subList = []
		tempChildList = []
		for row in n.rows:
			if row[maxGRIndex] is att:
				subList.append(row)
		child = Node(n, tempChildList, att, maxGRIndex, subList)
		n.childList.append(child)

	for child in n.childList:
		buildTree(child)

#Tests the rows of the test set by traversing the decision tree
def testRow(row, root):
	classified = False
	c = None
	current = root
	while classified is False:
		if row[0] is current.classification:
			classified = True
			c = row[0]
			break
		for child in current.childList:
			feat = child.feature
			if row[child.featIndex] is feat:
				current = child
				break

	if classified is True:
		return (1, c)
	else:
		return (0, c)


def run():

	if(len(sys.argv) < 4):
		sys.exit('Missing an argument. Arguments: Training Set file, Test Set File, Output File.')

	trainingFile = str(sys.argv[1]);
	testFile = str(sys.argv[2]);
	outputFile = str(sys.argv[3]);


	f = open(trainingFile, 'r')
	rows = [line.split() for line in f]
	numRecords = len(rows)
	numAtts = len(rows[1])
	dictList = []

	for x in range(0, numAtts):
		d = {}
		dictList.append(d)

	for row in rows:
		for x in range(0, numAtts):
			if(row[x] in dictList[x]):
				dictList[x][row[x]] += 1
			else:
				dictList[x][row[x]] = 1


	children = []
	n = Node(None, children, None, None, rows)
	buildTree(n)


	correct = 0
	f = open(testFile, 'r')
	f2 = open(outputFile, 'w')
	rows = [line.split() for line in f]
	for row in rows:
		value, c = testRow(row, n)
		correct += value
		s = 'Predicted Class: ' + str(row[0]) + '	Actual Class: ' + str(c)
		f2.write(s + '\n')


	accuracy = 100.0 * ((float(correct))/(float(len(rows))))
	s = 'Accuracy: ' + str(accuracy) + '%'
	f2.write(s)


run()
