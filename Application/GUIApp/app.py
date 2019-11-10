#--Imports------------------------------------------------------------------
import tkinter as tk
import json

import numpy as np
from keras.models import load_model

#---------------------------------------------------------------------------

#--Initialisations----------------------------------------------------------

filename = "log.txt"

fields = 'Day', 'Month', 'Year', 'Time', 'FestivalFactor'

output_fields = ['CrowdSize', 'MaleRatio']
other_fields = ['FemaleRatio']


inputs = [0.0] * len(fields)
outputs = []
others = []

model = None
modelPath = "trainedModel.h5"
#---------------------------------------------------------------------------

#--Form Functions----------------------------------------------------------------
#--1--
def makeform(root, fields):
	entries = []
	out_labels = []
	other_labels = []
	for field in fields:
		row = tk.Frame(root)
		lab = tk.Label(row, width=15, text=field, anchor='w')
		ent = tk.Entry(row)
		row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
		lab.pack(side=tk.LEFT)
		ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
		entries.append((field, ent))
	for outfield in output_fields:
		row = tk.Frame(root)
		lab = tk.Label(row, width=15, text=outfield, anchor='w')
		outlab = tk.Label(row, width=15, text="", anchor='w')
		row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
		lab.pack(side=tk.LEFT)
		outlab.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
		out_labels.append(outlab)
	for otherfield in other_fields:
		row = tk.Frame(root)
		lab = tk.Label(row, width=15, text=otherfield, anchor='w')
		otherlab = tk.Label(row, width=15, text="", anchor='w')
		row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
		lab.pack(side=tk.LEFT)
		otherlab.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
		other_labels.append(otherlab)
	return entries, out_labels, other_labels

#--2--
def fetch_inputs(entries):
	global inputs
	input_dict = {}
	index = 0
	for entry in entries:
		field = entry[0]
		text  = entry[1].get()
		input_dict[field] = float(text)
		inputs[index] = float(text)

		index += 1
	print(input_dict)

	global model
	global modelPath
	global outputs
	global others
	global filename
	global outlabs
	global otherlabs

	model = LoadModel(modelPath)
	outputs = PredictOutput(np.array([inputs]))[0]
	print("Inputs")
	print(inputs)
	print("Outputs")
	print(np.around(outputs, 2))

	# For Crowd Size and Male Ratio
	for out, lab in zip(outputs, outlabs):
		lab['text'] = str(round(out, 2))

	# For Female Ratio
	others.append(str(1 - outputs[1]))
	otherlabs[0]['text'] = str(round(1 - outputs[1], 2))

	LogData(inputs=inputs, outputs=outputs, others=others, filename=filename, option='a')


#--3--
def LogData(inputs=[], outputs=[], others=[], filename="log.txt", option='r'):
	if option == 'c': # Clear
		f = open(filename, 'w+')
		f.write("")
		f.close()
		return 1

	elif option == 'r': # Read
		f = open(filename, 'r')
		filestr = f.read()
		f.close()
		return filestr

	elif option == 'a': # Append
		global fields
		global output_fields
		global other_fields
		filestr = LogData(filename=filename, option='r')
		filestr += "Inputs:\n"
		for inp, field in zip(inputs, fields):
			filestr += "\t" + field + ": " + str(inp) + "\n"
		filestr += "Outputs:\n"
		for out, field in zip(outputs, output_fields):
			filestr += "\t" + field + ": " + str(out) + "\n"
		filestr += "Others:\n"
		for oth, field in zip(others, other_fields):
			filestr += "\t" + field + ": " + str(oth) + "\n"
		filestr += "\n\n"
		LogData(filename=filename, option='c')
		f = open(filename, 'w+')
		f.write(filestr)
		f.close()
		return 1

#---------------------------------------------------------------------------
#--Model Functions-----------------------------------------------------------
def LoadModel(modelPath="trainedModel.h5"):
	return load_model(modelPath)
def PredictOutput(inputs):
	global model
	if(model == None):
		return None
	return model.predict(inputs)

#---------------------------------------------------------------------------

#--Main Code----------------------------------------------------------------

root = tk.Tk()
ents, outlabs, otherlabs = makeform(root, fields)
root.bind('<Return>', (lambda event, e=ents: fetch_inputs(e)))   
b1 = tk.Button(root, text='Done',
	command=(lambda e=ents: fetch_inputs(e)))
b1.pack(side=tk.LEFT, padx=5, pady=5)
b2 = tk.Button(root, text='Quit', command=root.quit)
b2.pack(side=tk.LEFT, padx=5, pady=5)
root.mainloop()

