import matplotlib.pyplot as plt
import numpy as np
import re

def plotting(num_list, data, Pick = 0):
	assert len(num_list) == len(data), 'len(num_list) == len(data) should hold'
	assert Pick>= 0 and Pick<7, 'Pick>= 0 and Pick<7 should hold'
	Pick = 0
	plt.plot(num_list, data[:, Pick], label = 'Group' + chr(Pick + ord('A')))

	Pick = 2
	plt.plot(num_list, data[:, Pick], label = 'Group' + chr(Pick + ord('A')))

	Pick = 4
	plt.plot(num_list, data[:, Pick], label = 'Group' + chr(Pick + ord('A')))

	plt.ylabel('$log{P}$', fontsize = 20)
	plt.xlabel('Number of States', fontsize = 15)
	plt.legend(loc = 'best')
	

	plt.savefig('./probability.png')

	plt.show()
	return

def main():
	num_list = [5, 10, 20, 40, 80, 100]
	data = []
	###for num in [5, 10, 20, 40]:
	for num in num_list:
			filename = '../../probability/'+'prob_num{0}'.format(num)+'.txt'
			finalProb = []
			with open(filename, 'r') as file:
				for line in file:
					if 2 == len(line): continue
					lst = line.split()
					data.append(float(lst[-1]))
	data = np.array(data)
	data = data.reshape((len(num_list), 7))
	plotting(num_list, data)


if __name__ == "__main__":
    main()