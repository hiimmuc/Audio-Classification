from processing.audio_loader import loadWAV
from tqdm.auto import tqdm
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from pydub import AudioSegment
import soundfile as sf

from model import SpeakerEncoder, WrappedModel, ModelHandling
from utils import (read_config, cprint)
from server_utils import *
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, fbeta_score, roc_curve)

def evaluate_by_precision_recall(y_true, y_pred, beta_values=[1], target_names= ["Label '0'", "Label '1'"]):
    # get classification report
    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=5)

    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Now the normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # The diagonal entries are the accuracies of each class
    accuracy_per_classes = cm.diagonal()

    # calcualte f_beta
    fb_scores = {}
    for b in beta_values:
        fb_score = fbeta_score(y_true, y_pred, beta=b, pos_label=1, average=None)
        fb_scores[b] = fb_score

    return report, accuracy_per_classes, fb_scores

# check log folder exists
NAME_PATH = 'backup/8888/save/metadata/name_map.txt'
TEST_FOLDER = ''

# create test file

TEST_FILES_metadata = 'dataset/test_files.txt'
##
MODEL_PATH = str(
    Path('backup/8888/save/ResNetSE34/Softmax/model/best_state.pt'))
CONFIG_PATH = str(Path('yaml/configuration.yaml'))
args = read_config(CONFIG_PATH)
args = Namespace(**args)

# Load model
t0 = time.time()
net = WrappedModel(SpeakerEncoder(**vars(args), include_top=True)).to('cuda')
max_iter_size = args.step_size
speaker_model = ModelHandling(
    net, **dict(vars(args), T_max=max_iter_size))
speaker_model.loadParameters(MODEL_PATH, show_error=True)
speaker_model.__model__.eval()

print("Model Loaded time: ", time.time() - t0)
##
if __name__ == '__main__':
    with open(TEST_FILES_metadata, 'r') as rf:
        lines = [line.replace('\n', '') for line in rf.readlines()]
    with open(NAME_PATH, 'r') as rf:
        names = [line.replace('\n', '') for line in rf.readlines()]

    device = 'cuda'
    
    correct = 0
    labels = []
    preds = []
    with open('backup/8888/save/ResNetSE34/SoftmaxAngularProto/result/test_results.txt', 'w') as wf: 
        loader_bar = tqdm(
            lines, desc=">>>>Testing file: ", unit="files", colour="red")
        ##
        for line in loader_bar:
            label, fpath = line.split(' ')  # change pattern here
            audio = loadWAV(fpath,
                            args.audio_spec,
                            evalmode=True,
                            augment=False,
                            augment_options=[],
                            num_eval=1,
                            random_chunk=False)

            inp = torch.FloatTensor(audio).to('cuda')
            # print(speaker_model.predict(inp))
            outputs = (speaker_model.predict(inp)).softmax(dim=0).detach().cpu()
            prediction = outputs.max(0)

            correct += (int(prediction.indices) == int(label))
            
            preds.append(int(prediction.indices))
            labels.append(int(label))
            loader_bar.set_postfix(Precision=f"{round(correct/len(lines),3)}")
            wf.write(f"{label} {int(prediction.indices)} {fpath}\n")
            
    beta_values = [0.5, 2]
    prec_recall = evaluate_by_precision_recall(
        labels, preds, beta_values=beta_values, target_names = names)
    print("REPORT:\n", prec_recall[0])
    print("Accuracy for each class:")
    for i, cl in enumerate(prec_recall[1]):
        print(i, cl)
    for b in beta_values:
        print(f"F-{b}:", prec_recall[2][b])
    print("Precision:", correct/len(lines))
