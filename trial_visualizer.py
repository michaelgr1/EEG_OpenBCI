import os

import numpy as np
from matplotlib import pyplot as plt

import utils

LEFT_FREQ = 17

RIGHT_FREQ = 23

BANDWIDTH = 0.2

LEFT2 = 4

LEFT1 = 5

CENTER = 3

RIGHT1 = 1

RIGHT2 = 2

L_TO_R = [LEFT2, LEFT1, CENTER, RIGHT1, RIGHT2]

NAMES = {
	RIGHT2: "R2",
	RIGHT1: "R1",
	CENTER: "C",
	LEFT1: "L1",
	LEFT2: "L2"
}

WINDOW_SIZE = 3


def main():

	left_frequency_band = utils.FrequencyBand(LEFT_FREQ - BANDWIDTH / 2, LEFT_FREQ + BANDWIDTH / 2)
	right_frequency_band = utils.FrequencyBand(RIGHT_FREQ - BANDWIDTH / 2, RIGHT_FREQ + BANDWIDTH / 2)

	trial_directory = input("Trial Directory: ")

	path = os.getcwd() + "\\eeg_recordings\\" + trial_directory

	print("Loading data...")

	raw_data = utils.load_data(path)
	filter_settings = utils.FilterSettings(250, 13, 28)

	print("Slicing and filtering...")
	eeg_data, labels, fs, trial_classes = utils.slice_and_filter_data(path, filter_settings, raw_data)

	label_to_name = {}

	range_label = f"1-{len(labels)}"

	unique_labels = np.unique(np.array(labels))

	for label in unique_labels:
		for trial_class in trial_classes:
			if trial_class.label == label:
				label_to_name[label] = trial_class.name
				break

	while True:
		print("""Choose an option:
		1 - FFT
		2 - Bar Chart
		3 - All class FFT
		0 - Quit""")

		option = int(input())

		if option == 1:
			trial = input(f"Enter trial number {range_label}: ")

			index = int(trial) - 1
			current_eeg_data = eeg_data[index]
			plt.figure()

			for electrode in L_TO_R:
				extractor = current_eeg_data.feature_extractor(electrode - 1, fs)
				frequency, power = extractor.fft(WINDOW_SIZE)
				plt.plot(frequency, power, label=NAMES[electrode])

			plt.title("FFT for trial " + str(index + 1) + " of type " + str(label_to_name[labels[index]]))
			plt.legend()
			plt.show()

		elif option == 2:
			trial = input(f"Enter trial number {range_label}: ")
			index = int(trial) - 1
			current_eeg_data = eeg_data[index]

			plot_data = np.zeros((2, 5))

			i = 0

			for electrode in L_TO_R:
				extractor = current_eeg_data.feature_extractor(electrode - 1, fs)
				lp = extractor.average_band_amplitude(left_frequency_band, WINDOW_SIZE)
				rp = extractor.average_band_amplitude(right_frequency_band, WINDOW_SIZE)
				plot_data[0, i] = lp
				plot_data[1, i] = rp
				i += 1

			plt.figure()

			plt.title(f"Frequency Bar Chart for trial {index+1} of type {label_to_name[labels[index]]}")

			x = np.arange(5)

			plt.bar(x + 0.0, plot_data[0], width=0.4, label=f"{LEFT_FREQ} Hz")
			plt.bar(x + 0.5, plot_data[1], width=0.4, label=f"{RIGHT_FREQ} Hz")

			plt.xticks(x + 0.25, ("LEFT 2", "LEFT 1", "CENTER", "RIGHT 1", "RIGHT 2"))
			plt.ylabel("Band Amplitude")

			left_max = np.max(plot_data[0])
			right_max = np.max(plot_data[1])

			plt.plot(x, np.ones_like(x)*left_max, label=f"Max amplitude for {LEFT_FREQ} Hz")
			plt.plot(x, np.ones_like(x)*right_max, label=f"Max amplitude for {RIGHT_FREQ} Hz")

			plt.legend(loc="best")
			plt.show()
		elif option == 3:
			fig, ax = plt.subplots(len(label_to_name.keys()), len(L_TO_R))   # Labels by electrodes

			for i in range(unique_labels.shape[0]):
				label = unique_labels[i]
				class_name = label_to_name[label]

				for j in range(len(L_TO_R)):
					electrode = L_TO_R[j]

					ax[i, j].title.set_text(f"FFT for {NAMES[electrode]} of {class_name}")
					# ax[i, j].xlabel("Frequency Hz")
					# ax[i, j].ylabel("Amplitude")

					for k in range(len(eeg_data)):
						if labels[k] == label:
							extractor = eeg_data[k].feature_extractor(electrode - 1, fs)
							frequency, power = extractor.fft(WINDOW_SIZE)
							ax[i, j].plot(frequency[0:100], power[0:100])
			plt.show()
		elif option == 0:
			break


if __name__ == "__main__":
	main()
