# client
import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import requests
import simplejson
import soundfile as sf
import torch

from utils import cprint
from pydub import AudioSegment
from processing.wav_conversion import normalize_audio_amp


URL = "http://10.205.36.56:1234/quality"  # http://10.254.136.107:8111/

# from processing.wav_conversion
#
# def normalize_audio(signal):
#     try:
#         intinfo = np.iinfo(signal.dtype)
#         return signal / max( intinfo.max, -intinfo.min )

#     except ValueError: # array is not integer dtype
#         return signal / max( signal.max(), -signal.min())

def encode_audio(path):
    # audio, sr = sf.read(str(Path(path)))
    # segment -> np -> base64 -> b64 str
    sr = 8000
    audio_seg = AudioSegment.from_file(path)
    # convert to numpy
    audio = audio_seg.get_array_of_samples()
    audio = np.array(audio).astype(np.float64)
    audio = normalize_audio_amp(audio)
    # encode base64 str format
    audio_signal_bytes = base64.b64encode(audio)
    audio_signal_str = audio_signal_bytes.decode('utf-8')
    return audio_signal_str, sr

def get_response(audio):
    t = time.time()
    audio_encoded = encode_audio(audio)[0]
    
    data = {'callId': 'test_audio',
            'phone': '',
            'audio': audio_encoded}

    data_json = json.dumps(data)
    try:
        r = requests.post(URL, json=data_json)
        # print with color state of response
        print("Connection: ", end='')
        state = "Success" if int(r.status_code) == 200 else "Failed"
        color_text = 'g' if int(r.status_code) == 200 else 'r'
        cprint(text=state, fg=color_text, end=' ')

        response = r.json()
        print("Response time:", time.time() - t)

        quality = response["quality"]
        cprint(f"Quality:", fg='y', end=' ')
        cprint(str(quality), fg='g')
        print(response["percents"])
        

# print with color match state

    except Exception as e:
            print("Error when getting response ::: " + str(e))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TestService")
    parser.add_argument('--path',
                        type=str,
                        default=None,
                        help='path to file 2')
    args = parser.parse_args()
    t = time.time()
    print(f"<[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}]>")

    audio = 'dataset/test_samples/clean/wav/327905203/20220425192034-HNHYBIEF-134520_11502-10.61.85.11-vt_cskh_ohsyen6939_ccvpc-11502-05203-0000811905-327905203-CTM_0.wav'
    if args.path is None:
        audio = audio
    else:
        audio = args.path
    get_response(audio)
######################################################################