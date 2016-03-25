from graphviz import Digraph
import numpy as np

f = Digraph('Markov_Chain_Diagram', format = 'png')
f.body.extend(['rankdir=LR', 'size="10, 10"'])

A = np.loadtxt('../model/modelnhidden5groupAtrans.txt')

f.attr('node', shape='doublecircle')
f.node('START')
f.node('END')

DIM = A.shape[0] - 2

f.attr('node', shape='circle')

for it in range(DIM):
	if(A[-2,it]>1e-4): # plot the edge only the probability is greater than 1e-4
		f.edge('START', '{0}'.format(it), label= str(round(A[-2, it], 3)))

for it in range(DIM):
	if(A[it,-1]>1e-4):
		f.edge('{0}'.format(it), 'END', label= str(round(A[it, -1], 3)))

for outer_it in range(DIM):
		for inner_it in range(DIM):
			if(A[outer_it,inner_it]>1e-4):
				f.edge('{0}'.format(outer_it), '{0}'.format(inner_it), label= str(round(A[outer_it,inner_it], 3)))

f.render(filename = 'HiddenMarkov{0}'.format(DIM), view = True)
