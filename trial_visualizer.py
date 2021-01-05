import os

import numpy as np
from matplotlib import pyplot as plt

import utils

LEFT_FREQ = 20

RIGHT_FREQ = 24

BANDWIDTH = 0.2

LEFT2 = 5

LEFT1 = 4

CENTER = 3

RIGHT1 = 2

RIGHT2 = 1

L_TO_R = [LEFT2, LEFT1, CENTER, RIGHT1, RIGHT2]

NAMES = {
	RIGHT2: "R2",
	RIGHT1: "R1",
	CENTER: "C",
	LEFT1: "L1",
	LEFT2: "L2"
}

WINDOW_SIZE = 2.048


def main():

	left_frequency_band = utils.FrequencyBand(LEFT_FREQ - BANDWIDTH / 2, LEFT_FREQ + BANDWIDTH / 2)
	right_frequency_band = utils.FrequencyBand(RIGHT_FREQ - BANDWIDTH / 2, RIGHT_FREQ + BANDWIDTH / 2)

	trial_directory = input("Trial Directory: ")

	path = os.getcwd() + "\\eeg_recordings\\" + trial_directory

	print("Loading data...")

	raw_data = utils.load_data(path)
	filter_settings = utils.FilterSettings(250, 15, 35, reference_electrode=3)

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
	print("Loading done")

	while True:
		print("""Choose an option:
		1 - FFT
		2 - Bar Chart
		3 - All class FFT
		4 - Statistics
		5 - FFT Everything
		6 - Class Average
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

					f17_average = utils.AccumulatingAverage()
					f23_average = utils.AccumulatingAverage()

					for k in range(len(eeg_data)):
						if labels[k] == label:
							extractor = eeg_data[k].feature_extractor(electrode - 1, fs)
							frequency, power = extractor.fft(WINDOW_SIZE)
							f17 = extractor.frequency_amplitude(LEFT_FREQ, WINDOW_SIZE)
							f23 = extractor.frequency_amplitude(RIGHT_FREQ, WINDOW_SIZE)
							f17_average.add_value(f17)
							f23_average.add_value(f23)
							ax[i, j].plot(frequency[0:100], power[0:100])

					t = np.linspace(0, 40, 2)
					ax[i, j].plot(t, np.ones_like(t) * f17_average.compute_average(), label="17 Hz average")
					ax[i, j].plot(t, np.ones_like(t) * f23_average.compute_average(), label="23 Hz average")
					ax[i, j].legend()
			plt.show()
		elif option == 4:

			correct_ssl_count = 0
			correct_ssr_count = 0

			ssl_sum = 0
			ssr_sum = 0

			correct_indexes = []

			for i in range(len(eeg_data)):
				l2_extractor = eeg_data[i].feature_extractor(LEFT2 - 1, fs)
				l1_extractor = eeg_data[i].feature_extractor(LEFT1 - 1, fs)
				r1_extractor = eeg_data[i].feature_extractor(RIGHT1 - 1, fs)
				r2_extractor = eeg_data[i].feature_extractor(RIGHT2 - 1, fs)

				if label_to_name[labels[i]] == "SS-L":
					amplitudes = [
						l2_extractor.peak_band_amplitude(left_frequency_band, WINDOW_SIZE),
						l1_extractor.peak_band_amplitude(left_frequency_band, WINDOW_SIZE),
						r1_extractor.peak_band_amplitude(left_frequency_band, WINDOW_SIZE),
						r2_extractor.peak_band_amplitude(left_frequency_band, WINDOW_SIZE)
					]

					max_amplitude = max(amplitudes)

					for j in range(len(amplitudes)):
						if amplitudes[j] == max_amplitude:
							if j == 2 or j == 3:  # max amplitude on r1 or r2
								correct_ssl_count += 1
								correct_indexes.append(i)
					ssl_sum += 1
				elif label_to_name[labels[i]] == "SS-R":
					amplitudes = [
						l2_extractor.peak_band_amplitude(right_frequency_band, WINDOW_SIZE),
						l1_extractor.peak_band_amplitude(right_frequency_band, WINDOW_SIZE),
						r1_extractor.peak_band_amplitude(right_frequency_band, WINDOW_SIZE),
						r2_extractor.peak_band_amplitude(right_frequency_band, WINDOW_SIZE)
					]

					max_amplitude = max(amplitudes)

					for j in range(len(amplitudes)):
						if amplitudes[j] == max_amplitude:
							if j == 0 or j == 1:  # max amplitude on l2 or l1
								correct_ssr_count += 1
								correct_indexes.append(i)
					ssr_sum += 1

			print("From SS-L {}% of trials have higher amplitude in electrodes R2 or R1".format(correct_ssl_count / ssl_sum * 100))
			print("From SS-R {}% of trials have higher amplitude in electrodes L2 or L1".format(correct_ssr_count / ssr_sum * 100))

			print(correct_indexes)
		elif option == 5:

			all_data = utils.EegData()

			for i in range(len(eeg_data)):
				all_data.append_data(eeg_data[i].to_row_array())

			plt.figure()

			data_row_array = all_data.to_row_array()

			win_size_in_samples = int(WINDOW_SIZE * 250)

			for electrode in L_TO_R:
				extractor = all_data.feature_extractor(electrode - 1, fs)
				freq, power = extractor.fft(WINDOW_SIZE)

				data = data_row_array[electrode - 1, :]

				plt.plot(freq, power, label=NAMES[electrode] + " - Custom Welch")

				# power, freq = DataFilter.get_psd_welch\
				# 	(data, win_size_in_samples, int(win_size_in_samples / 2), 250, WindowFunctions.HAMMING.value)
				#
				# plt.plot(freq, power, label=NAMES[electrode] + " - Library Welch")

			plt.xlabel("Frequency (Hz)")
			plt.ylabel("Amplitude")
			plt.title("Re-referenced FFT for entire trial")
			plt.legend(loc="best")

			plt.show()
		elif option == 6:
			averages = {}

			for label in np.unique(labels):
				averages[label] = {}
				for electrode in L_TO_R:
					averages[label][electrode] = utils.AccumulatingAverages()

			frequency = None

			for i in range(len(eeg_data)):
				for electrode in L_TO_R:
					extractor = eeg_data[i].feature_extractor(electrode - 1, fs)
					frequency, power = extractor.fft(WINDOW_SIZE)
					averages[labels[i]][electrode].add_values(power)

			plt.figure()

			for label in np.unique(labels):
				label_str = label_to_name[label]
				for electrode in L_TO_R:
					name = NAMES[electrode]
					plt.plot(frequency, averages[label][electrode].compute_average(), label=f"{label_str} {name} average")

			plt.title("Class Averages")
			plt.xlabel("Frequency")
			plt.ylabel("Amplitude")

			plt.legend(loc="best")
			plt.show()

		elif option == 0:
			break


if __name__ == "__main__":
	main()
