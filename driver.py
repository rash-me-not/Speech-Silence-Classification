'''
driver assignment 2
Name:  
'''
from sklearn.mixture import GaussianMixture as gaussian
from mydsp import audioframes, rmsstream
import numpy as np
import matplotlib.pyplot as plt


def speech_silence(filename):
    """speech_silence(filename)
    Given speech file filename, train a 2 mixture GMM on the
    RMS intensity and label each frame as speech or silence.
    Provides a plot of results.
    """

    audio = audioframes.AudioFrames(filename,10,20)

    rms = rmsstream.RMSStream(audio)

    rms_val = []
    for i in rms:
        rms_val.append(i)

    rms_reshaped = np.reshape(rms_val,(-1, 1))

    gmm = gaussian(n_components=2).fit(rms_reshaped)

    labels = gmm.predict(rms_reshaped)

    silence_index = []
    speech_index = []
    for i in range(0,len(labels),1):
        if labels[i] == 0:
            silence_index.append(i)
        else:
            speech_index.append(i)

    silence_val = np.asarray(rms_val)[silence_index]
    speech_val = np.asarray(rms_val)[speech_index]


    silence_index = np.asarray(silence_index) * audio.get_frameadv_ms() * 0.001
    speech_index = np.asarray(speech_index) * audio.get_frameadv_ms() * 0.001

    plt.figure()
    tidx = np.arange(len(audio)) * audio.get_frameadv_ms() * 0.001
    plt.plot(tidx, rms_val)
    plt.scatter(silence_index,np.asarray(silence_val), marker= 'o',edgecolors='green', label='Silence')
    plt.scatter(speech_index, np.asarray(speech_val), marker='x',edgecolors='yellow', label = 'Speech')
    plt.legend()
    plt.title("RMS Intensity in dB rel for audio 'shaken.wav'")
    plt.xlabel("Time (sec)")
    plt.ylabel("dB Rel (rms value)")
    plt.show()

if __name__ == '__main__':
    # If we are here, we are in the script-level environment; that is
    # the user has invoked python driver.py.  The module name of the top
    # level script is always __main__

    speech_silence("shaken.wav")
