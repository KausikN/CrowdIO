import numpy as np
import librosa
import os
import glob

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
	print ("Started")
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
					logspec = librosa.amplitude_to_db(melspec)		# MISSING LOGAMPLITUDE
					logspec = melspec
					logspec = logspec.T.flatten()[:, np.newaxis].T
					log_specgrams.append(logspec)
					labels.append(label)
			print (PTJ + ITJ + 270, ITJ)
			PTJ = PTJ + 1
		ITJ = ITJ + 1
	log_specgrams = np.array(log_specgrams)
	print ("Done")
	print (log_specgrams.shape)
	log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames)
	features = log_specgrams
	return np.array(features)

features = extract_features("C:/GitHub Codes and Projects/Projects", ["CrowdIO-Project"], "0.wav")
print("\n\n", features.shape, " - Features:\n\n", features)