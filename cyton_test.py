from brainflow import BoardShim, BoardIds, BrainFlowInputParams, LogLevels
import time


def main():
	BoardShim.enable_board_logger()
	BoardShim.enable_dev_board_logger()

	params = BrainFlowInputParams()
	# params.serial_port = "COM3"

	board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)

	board.log_message(LogLevels.LEVEL_INFO.value, "Preparing session...")

	board.prepare_session()

	board.log_message(LogLevels.LEVEL_INFO.value, "Starting stream...")

	board.start_stream()

	start_time = time.time()

	run_time = 5  # 5 seconds

	board.log_message(LogLevels.LEVEL_INFO.value, "Start time = {}".format(start_time))

	while time.time() - start_time < run_time:
		if board.get_board_data_count() > 0:
			data = board.get_board_data()

			print("Got data, time = {}, data count = {}.".format(time.time() - start_time, len(data)))

			# print(data[BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)])

		time.sleep(0.001)

	board.log_message(LogLevels.LEVEL_INFO.value, "Stopping stream...")
	board.stop_stream()
	board.release_session()


if __name__ == "__main__":
	main()
