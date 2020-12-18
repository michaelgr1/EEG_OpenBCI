import math

import matplotlib.pylab as plt
import numpy as np
import padasip as pa

t = np.linspace(0, 2*math.pi, 1000)
signal = np.sin(5 * t)
noise = 0.1 * np.random.randn(t.size)

noisy_signal = signal + noise

lms_filter = pa.filters.FilterLMS(1)

y, e, w = lms_filter.run(noisy_signal, signal.reshape(-1, 1))


plt.figure()
plt.plot(t, signal, label="Original Signal")
plt.plot(t, noisy_signal, label="Noisy Signal")
plt.plot(t, y, label="Filter Output")
plt.plot(t, noise, label="noise")
plt.plot(t, (noisy_signal - y), label="Noisy Minus Output")

plt.legend(loc="best")
plt.show()

