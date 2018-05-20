import numpy as np
from scipy import *
from scipy.signal import decimate
import scipy.io.wavfile
import sys


def main(arg):
    try:
        w, signal = scipy.io.wavfile.read(arg)
    except:
        print('ERROR: File "{0}" not found'.format(arg))
        return
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    frame_size = 10000
    shift = 6000
    window = np.hamming(frame_size)
    harmonic = 4
    output = [0, 0]
    for i in range(0, len(signal), shift):
        current_fragment = signal[i:i + frame_size]
        current_fragment = [c * w for c, w in zip(current_fragment, window)]
        curr_fr_rfft = np.fft.rfft(current_fragment)
        freq = np.fft.rfftfreq(len(window), 1. / w)
        res = curr_fr_rfft.copy()
        after_decimate = [0] * harmonic
        for l in range(2, harmonic + 2):
            after_decimate[l-2] = curr_fr_rfft[::l]
        for l in after_decimate:
            for j in range(len(l)):
                res[j] *= l[j]
        fMax = freq[np.argmax(res[2:])]
        if 260 >= fMax >= 160:
            output[1] += 1
        elif 160 > fMax >= 75:
            output[0] += 1
    if output[1] >= output[0]:
        gender = 'K'
    elif output[0] > output[1]:
        gender = 'M'

    print(gender)
    return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('ERROR: invalid amount of parameters')
    else:
        main(sys.argv[1])
