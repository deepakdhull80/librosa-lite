import os
from datetime import datetime
import soundfile

from librosa_lite.core import spectrogram, mel_filter, log_mel_spec, mel_to_db

def timeit(f):
    def run():
        st = datetime.now()
        res = f()
        print(f"Total Time Taken: {datetime.now() - st}")
        return res
    return run

@timeit
def test_spectrogram():
    file_path = "data\OSR_us_000_0010_8k.wav"
    assert os.path.exists(file_path), f"File not found{file_path}, This file is mendatory for this unit test.\n Download file from here http://www.voiptroubleshooter.com/open_speech/american.html"

    wav, sr = soundfile.read(file_path)
    
    filters = mel_filter(128, sr, 512)
    
    spec = spectrogram(wav, sr=sr)
    assert spec.shape == (3360,128), f"Failed: spectrogram shape should be expected (3360,128) and actual ({spec.shape}) "
    print("mel_spec: ",spec.shape)
    
    spec = mel_to_db(log_mel_spec(spec))
    
    print(f"Finished, All test cases passed.")

if __name__ == '__main__':
    test_spectrogram()