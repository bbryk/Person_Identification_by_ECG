
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

import numpy as np


def detect_peaks(signal, sampling_rate, to_plot=False):
    def suppress_neighborhood_peaks(signal, indices, neighborhood_size=50):
        sorted_indices = sorted(indices, key=lambda i: signal[i], reverse=True)

        final_peaks = []

        while sorted_indices:
            current_peak = sorted_indices.pop(0)
            final_peaks.append(current_peak)

            sorted_indices = [i for i in sorted_indices if abs(i - current_peak) > neighborhood_size]

        return np.array(final_peaks)

    def apply_operator(y1, m=5):
        N = len(y1)
        y2 = np.zeros(N)
        for n in range(N):
            window_sum = np.sum(y1[max(0, n - m):min(N, n + m + 1)] ** 2)
            y2[n] = y1[n] * window_sum
        return y2

    el_id = 1

    # filtered324 = np.load(f"filtered_ecg/{i}/filtered_{i}.npy")
    # fs = 428

    import numpy as np
    filtered324 = signal
    # filtered324 = np.load("filtered_real3.npy")[1000:-1000]
    sampling_rate = sampling_rate
    fs = sampling_rate

    signal = filtered324

    lowcut = 10
    highcut = 25
    order = 4

    b, a = butter(order, [lowcut, highcut], btype='band', fs=fs)

    filtered_signal = filtfilt(b, a, signal)

    y1 = filtered_signal

    y2 = apply_operator(y1)

    import numpy as np
    from scipy.signal import find_peaks

    threshold = max(y2) / 4
    print(threshold)

    signal = y2

    peaks, properties = find_peaks(signal, height=threshold)
    print(peaks)


    signal = y2
    indices = peaks

    suppressed_peaks = suppress_neighborhood_peaks(signal, indices, neighborhood_size=int(sampling_rate / 10))

    sorted_peaks = sorted(suppressed_peaks)

    import matplotlib.pyplot as plt
    import numpy as np

    signal = filtered324
    indices = sorted_peaks

    if to_plot == True:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        axes[0].plot(signal, label='Signal')
        axes[0].set_title('Original Signal')
        axes[0].set_xlabel('Counts')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_ylim(0, 1)

        axes[1].plot(y1, label='Filtered Signal', color='green')
        axes[1].set_title('Filtered Signal with bandpass for QRS complex')
        axes[1].set_xlabel('Counts')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_ylim(-2*max(y1), 2*max(y1))  # Set y-axis range for consistency

        axes[2].plot(y2, label='Processed Signal', color='red')
        axes[2].set_title('Processed Signal with small-value suppression operator')
        axes[2].set_xlabel('Counts')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_ylim(-2*max(y2), 2*max(y2))

        for ax in axes:
            for index in indices:
                if ax == axes[0]:
                    ax.axvline(x=index, color='orange', linestyle='--', label='Peaks')


        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.savefig('figures/812_rpeak.png', dpi=300)
        plt.show()
    print(indices)
    return indices






