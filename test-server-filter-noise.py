import base64
import enum
import io
import json
import os
import time
from argparse import Namespace
from json import dumps
from pathlib import Path

import numpy as np
import torch
from flask import (Flask, Markup, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from flask_restful import Api, Resource
from pydub import AudioSegment
from werkzeug.utils import secure_filename
import soundfile as sf
import torch
import torch.nn as nn

from model import SpeakerEncoder, WrappedModel, ModelHandling
from utils import (read_config, cprint)
from processing.audio_loader import loadWAV
from server_utils import *

# check log folder exists
log_service_root = str(Path('log_service/'))
os.makedirs(log_service_root, exist_ok=True)

# ==================================================load Model========================================

NAME_PATH = 'backup/8888/save/metadata/name_map.txt'
name_map = {}
with open(NAME_PATH) as rf:
    for line in rf.readlines():
        v, k = line.replace('\n', '').split()
        name_map[int(k)] = v

# create test file

TEST_FILES_metadata = 'dataset/test_files.txt'
##
MODEL_PATH = str(
    Path('backup/8888/save/ResNetSE34/SoftmaxAngularProto/model/best_state.pt'))
CONFIG_PATH = str(Path('yaml/configuration.yaml'))
args = read_config(CONFIG_PATH)
args = Namespace(**args)

# Load model
sr = args.audio_spec['sample_rate']
t0 = time.time()
net = WrappedModel(SpeakerEncoder(**vars(args), include_top=True)).to('cuda')
max_iter_size = args.step_size
speaker_model = ModelHandling(
    net, **dict(vars(args), T_max=max_iter_size))
speaker_model.loadParameters(MODEL_PATH, show_error=True)
speaker_model.__model__.eval()

print("Model Loaded time: ", time.time() - t0)
##

# ================================================Flask API=============================================
# Set up env for flask
app = Flask(__name__, template_folder='templates')
app.secret_key = 'super secret key'

app.config['UPLOAD_FOLDER'] = log_service_root
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['DEBUG'] = True

api = Api(app)

# for matching call
@app.route('/quality', methods=['POST'])
def check_quality():
    audio_data = None
    
    current_day = str(time.strftime('%Y-%m-%d', time.gmtime())).replace('-', '').replace(' ', '_').replace(':', '')
    # create log dir
    log_audio_path = str(Path(f'log_service/{current_day}/audio'))
    os.makedirs(log_audio_path, exist_ok=True)
    log_results_path = str(Path(f'log_service/{current_day}/results'))
    os.makedirs(log_results_path, exist_ok=True)
    log_audio_path_id = os.path.join(log_audio_path, "unknown_number")
    os.makedirs(log_audio_path_id, exist_ok=True)
    #
    cprint(text=f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}]", fg='k', bg='g')
    ####################
    # Get request
    t0 = time.time()
    json_data = request.get_json()
    
    # print("\n> JSON <", json_data)
    if 'audio' in json_data:
        data_json = json.loads(json_data)
        call_id = data_json['callId']
        phone = data_json['phone']
        audio_data = data_json["audio"]

        print("Got audio signal in", time.time() - t0, 'sec', end=' || ')
        
        # create dir to save
        phone = "unknown_number" if (len(phone)==0) else phone
        log_audio_path_id = os.path.join(log_audio_path, phone)
        log_result_id =  os.path.join(log_results_path, phone)
        
        os.makedirs(log_audio_path_id, exist_ok = True)
        os.makedirs(log_result_id, exist_ok = True)
    else:
        raise "Error: no data provide"
        
    print("Phone number:", phone, end=' || ')
    
    # convertstring of base64 to np array
    dtype = np.float64
    audio_data_np = decode_audio(audio_data, args.audio_spec, dtype) 
    
    ####################
    # save log audio 
    print("Saving audio files...")   
    print(f"> Log files:")
    log_audio_path_id_log = os.path.join(log_audio_path_id, 'logs')
    os.makedirs(log_audio_path_id_log, exist_ok=True)
    
    if not any(str(call_id) in fname for fname in os.listdir(log_audio_path_id_log)):
        # check whether ref audio is exists
        save_path = os.path.join(log_audio_path_id_log, f'{call_id}.wav')
        sf.write(save_path, audio_data_np, sr)
    
    ####################
    audio = loadWAV(audio_data_np,
                    args.audio_spec,
                    evalmode=True,
                    augment=False,
                    augment_options=[],
                    num_eval=1,
                    random_chunk=False)
    inp = torch.FloatTensor(audio).to('cuda')
    print(speaker_model.predict(inp))
    outputs = (speaker_model.predict(inp)).softmax(dim=-1).detach().cpu()
    
    prediction = int(outputs.max(0).indices)
    audio_type = name_map[prediction]
    
    return jsonify({"quality": str(audio_type), "percents": ' '.join([str(el) for el in outputs.squeeze().tolist()])})




@app.route('/', methods=['GET'])
def get_something():
    """dump script"""
    pass


if __name__ == '__main__':
    #     app.run(debug=True)
    app.run(debug=False, host='0.0.0.0', port=8211)
