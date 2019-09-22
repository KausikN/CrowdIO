import numpy as np
import librosa
import os
import glob
import matplotlib.pyplot as plt
import pylab

def windows(data, window_size):
	start = 0
	while start < len(data):
		yield start, start + window_size
		start += (window_size / 2)

def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands = 128, frames = 128):
	window_size = 512*127
	log_specgrams = []
	labels = []
	ITJ = 0
	print ("Started Extracting Features...")
	for l, sub_dir in enumerate(sub_dirs):
		print ("In subdir: ", sub_dir)
		PTJ = 1
		for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
			sound_clip, s = librosa.load(fn)
			label = ITJ
			for(start, end) in windows(sound_clip, window_size):
				if(len(sound_clip[int(start):int(end)]) == window_size):
					signal = sound_clip[int(start):int(end)]
					melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
					logspec = librosa.amplitude_to_db(melspec)
					logspec = logspec.T.flatten()[:, np.newaxis].T
					log_specgrams.append(logspec)
					labels.append(label)
			print (PTJ + ITJ + 270, ITJ)
			PTJ = PTJ + 1
		ITJ = ITJ + 1
	log_specgrams = np.array(log_specgrams)
	print ("Finished Extracting Features")
	print (log_specgrams.shape)
	log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames)
	features = log_specgrams
	return np.array(features), labels

path = input("Enter Path of subdirectories: ")
subdirs = input("Enter Sub Directories: ").split(',')
features, labels = extract_features(path, subdirs, "*.wav")
print("\n\n", len(labels), " - Labels:\n\n", labels)
print("\n\n", features.shape, " - Features:\n\n", features)

# Plotting spectrograms
subplot_check = input("Do you want subplots: ")

if subplot_check == "Y":
	n_columns = int(input("Enter no of columns in plot: "))
	fig, axs = plt.subplots(int(features.shape[0]/n_columns) + 1, n_columns)

	i = 0
	j = 0 
	for imgindex in range(features.shape[0]):
		axs[i, j].imshow(features[imgindex], cmap=pylab.get_cmap('YlGnBu'))
		axs[i, j].set_title("Axis [" + str(i) + ", " + str(j) + "]")
		j = j + 1
		if j == n_columns:
			j = 0
			i = i + 1

	for ax in axs.flat:
		ax.set(xlabel="bands", ylabel="frames")

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axs.flat:
		ax.label_outer()
else:
	for imgindex in range(features.shape[0]):
		fig = plt.figure(imgindex)
		plt.imshow(features[imgindex], cmap=pylab.get_cmap('YlGnBu'))

plt.show()