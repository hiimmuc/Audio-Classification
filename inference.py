import csv
import glob
import os
import random
import subprocess
import sys
import time
from math import fabs
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import SpeakerEncoder, WrappedModel, ModelHandling

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, fbeta_score, roc_curve)
from tqdm import tqdm
from utils import ComputeErrorRates, ComputeMinDcf, tuneThresholdfromScore

# ---------------------------------//
# Evaluation code - must run on single GPU


def inference(args):
    net = WrappedModel(SpeakerEncoder(**vars(args)))
       
    max_iter_size = args.step_size
    
    speaker_model = ModelHandling(
        net, **dict(vars(args), T_max=max_iter_size))

    model_save_path = os.path.join(
        args.save_folder, f"{args.model['name']}/{args.criterion['name']}/model")

    result_save_path = os.path.join(
        args.save_folder, f"{args.model['name']}/{args.criterion['name']}/result")

    # Write args to score_file
    settings_file = open(result_save_path + '/settings.txt', 'a+')
    score_file = open(result_save_path + "/Inference_log.txt", "a+")
    test_log_file = open(result_save_path + "/Testing_log.txt", "a+")
    # summary settings
    settings_file.write(
        f'\n[INFER]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
    score_file.write(
        f'\n[INFER]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
    # write the settings to settings file
    for items in vars(args):
        # print(items, vars(args)[items])
        settings_file.write('%s %s\n' % (items, vars(args)[items]))
    settings_file.flush()
    settings_file.close()

    # priority: define weight -> best weight -> last weight
    if args.initial_model_infer:
        chosen_model_state = args.initial_model_infer
    elif os.path.exists(f'{model_save_path}/best_state.pt'):
        chosen_model_state = f'{model_save_path}/best_state.pt'
    else:
        model_files = glob.glob(os.path.join(
            model_save_path, 'model_state_*.pt'))
        chosen_model_state = model_files[-1]

    # duplicate best state to avoid missing best
    if 'best_state' in chosen_model_state and args.eval is True:
        ver = 0
        copy_name = chosen_model_state.split(
            '.')[0] + f'_copy_{ver}' + '.' + chosen_model_state.split('.')[-1]
        while os.path.exists(copy_name):
            ver += 1
            copy_name = chosen_model_state.split(
                '.')[0] + f'_copy_{ver}' + '.' + chosen_model_state.split('.')[-1]
        subprocess.call(f"cp {chosen_model_state} {copy_name}", shell=True)
    else:
        copy_name = None

    print(f'Loading model from {chosen_model_state}')
    speaker_model.loadParameters(chosen_model_state)
    speaker_model.__model__.eval()

    # set defalut threshold
    threshold_set = args.test_threshold
    scoring_mode = args.scoring_mode
    num_eval = args.num_eval

    # Evaluation from list
    if args.eval is True:

        sc, lab, trials = speaker_model.evaluateFromList(
            listfilename=args.evaluation_file,
            distributed=args.distributed,
            dataloader_options=args.dataloader_options,
            cohorts_path=args.cohorts_path,
            num_eval=args.num_eval,
            scoring_mode=scoring_mode)

        target_fa = np.linspace(5, 0, num=50)

        # results['gmean'] = [idxG, gmean[idxG], thresholds[idxG]]
        # results['roc'] = [tunedThreshold, eer, metrics.auc(fpr, tpr), optimal_threshold]
        # results['prec_recall'] = [precision, recall, fscore[ixPR], thresholds_[ixPR]]

        result = tuneThresholdfromScore(sc, lab, target_fa)

        ####
        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, threshold = ComputeMinDcf(
            fnrs, fprs, thresholds, args.dcf['dcf_p_target'], args.dcf['dcf_c_miss'], args.dcf['dcf_c_fa'])
        ####

        # print('tfa [thre, fpr, fnr]')
        best_sum_rate = 999
        best_tfa = None
        for i, tfa in enumerate(target_fa):
            # print(tfa, result[0][i])
            sum_rate = result['roc'][0][i][1] + result['roc'][0][i][2]
            if sum_rate < best_sum_rate:
                best_sum_rate = sum_rate
                best_tfa = result['roc'][0][i]

        print("\n[RESULTS]\nROC:",
              f"Best sum rate {best_sum_rate} at {best_tfa}, AUC {result['roc'][2]}\n",
              f">> EER {result['roc'][1]}%  min-DCF {mindcf:.5f} at threshold {result['roc'][-1]}\n",
              f">> Gmean result: \n>>> EER: {(1 - result['gmean'][1]) * 100}% at threshold {result['gmean'][2]}\n>>> ACC: {result['gmean'][1] * 100}%\n",
              f">> F-score {result['prec_recall'][2]}% at threshold {result['prec_recall'][-1]}\n")

        score_file.writelines(
            [f"[Evaluation] result on: [{args.evaluation_file}] with [{args.initial_model_infer}]\n",
             f"Best sum rate {best_sum_rate} at {best_tfa}\n",
             f" EER {result['roc'][1]}%  min-DCF {mindcf:.5f} at threshold {result['roc'][-1]}\nAUC {result['roc'][2]}\n",
             f"Gmean result:\n",
             f" EER: {(1 - result['gmean'][1]) * 100}% at threshold {result['gmean'][2]}\n>>> ACC: {result['gmean'][1] * 100}%\n=================>\n"])
        score_file.close()

        # write to file
        write_file = Path(result_save_path, 'evaluation_results.csv')
        fa_pairs = []
        fr_pairs = []
        
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(
                ['audio_1', 'audio_2', 'label', 'predict_label', 'score'])
            preds = []
            for score, label, pair in zip(sc, lab, trials):
                pred = int(score >= result['gmean'][2])
                com, ref = pair.strip().split(' ')
                spamwriter.writerow([com, ref, label, pred, score])
                preds.append(pred)
                if int(label) == 1 and int(label) != pred:
                    # False rejected
                    fr_pairs.append(f"{com},{ref},{score}\n")
                if int(label) == 0 and int(label) != pred:
                    # False accepted
                    fa_pairs.append(f"{com},{ref},{score}\n")                

            # print out metrics results
            beta_values = [0.5, 2]
            prec_recall = evaluate_by_precision_recall(
                lab, preds, beta_values=beta_values)
            print("REPORT:\n", prec_recall[0])
            print("Accuracy for each class:",
                  f"\n0's: {prec_recall[1][0]}\n1's: {prec_recall[1][1]}")
            for b in beta_values:
                print(f"F-{b}:", prec_recall[2][b])
                
        ## print out false pairs
        with open(Path(result_save_path, 'false_rejected_pairs.txt'), 'w') as wf:
            wf.writelines(fr_pairs)
        with open(Path(result_save_path, 'false_accepted_pairs.txt'), 'w') as wf:
            wf.writelines(fa_pairs)

        # decide wether keep the current state
        if copy_name:
            keep_file = str(
                input(f"Keep this version? '{copy_name}' (Y/N): ")).lower()
            if keep_file.strip().lower() == 'n':
                subprocess.call(f"rm {copy_name}", shell=True)
                print("removed copy file...")
        print("\n=============END===============//\n")
        sys.exit(1)

    # Test from list (audio1,audio2) and compare to truth file
    if args.test is True:
        if threshold_set == 0:
            print("Threshold is undefined, run evaluation first...")
            sc, lab, trials = speaker_model.evaluateFromList(
                listfilename=args.evaluation_file,
                distributed=args.distributed,
                dataloader_options=args.dataloader_options,
                cohorts_path=args.cohorts_path,
                num_eval=args.num_eval,
                scoring_mode=scoring_mode)

            target_fa = np.linspace(5, 0, num=50)

            result = tuneThresholdfromScore(sc, lab, target_fa)
            ####

            best_sum_rate = 999
            best_tfa = None
            for i, tfa in enumerate(target_fa):
                sum_rate = result['roc'][0][i][1] + result['roc'][0][i][2]
                if sum_rate < best_sum_rate:
                    best_sum_rate = sum_rate
                    best_tfa = result['roc'][0][i]

            print("\n[RESULTS]\nROC:",
                  f"Best sum rate {best_sum_rate} at {best_tfa}, AUC {result['roc'][2]}\n",
                  f">> EER {result['roc'][1]}% at threshold {result['roc'][-1]}\n",
                  f">> Gmean result: \n>>> EER: {(1 - result['gmean'][1]) * 100}% at threshold {result['gmean'][2]}\n>>> ACC: {result['gmean'][1] * 100}%\n",
                  f">> F-score {result['prec_recall'][2]}% at threshold {result['prec_recall'][-1]}\n")
            threshold_set =  result['gmean'][-1]
            
        speaker_model.testFromList(args.verification_file,
                                   thresh_score=threshold_set,
                                   output_file=args.log_test_files['com'],
                                   distributed=args.distributed,
                                   dataloader_options=args.dataloader_options,
                                   cohorts_path=args.cohorts_path,
                                   num_eval=args.num_eval,
                                   scoring_mode=scoring_mode)

        roc, prec_recall = evaluate_result(path=args.log_test_files['com'], ref=args.log_test_files['ref'])
        test_log_file.writelines([f">{time.strftime('%Y-%m-%d %H:%M:%S')}<",
                                  f"Test result on: [{args.verification_file}] with [{args.initial_model_infer}]\n",
                                  f"Threshold: {threshold_set}\n",
                                  f"ROC: {roc}\n",
                                  f"Report: \n{prec_recall}\n",
                                  f"Save to {args.log_test_files['com']} and {args.log_test_files['ref']} \n========================================\n"])
        test_log_file.close()
        sys.exit(1)

    # Prepare embeddings for cohorts/verification
    if args.prepare is True:
        speaker_model.prepare(eval_frames=args.eval_frames,
                      source=args.train_annotation,
                      save_path=args.cohorts_path,
                      num_eval=num_eval,
                      prepare_type=args.prepare_type)
        sys.exit(1)

    # Predict
    if args.predict is True:
        """
        Predict new utterance based on distance between its embedding and saved embeddings.
        """
        embeds_path = Path(args.save_path, 'embeds.pt')
        classes_path = Path(args.save_path, 'classes.npy')
        embeds = torch.load(embeds_path).to(torch.device(args.device))
        classes = np.load(str(classes_path), allow_pickle=True).item()

        if args.test_list.endswith('.txt'):
            files = []
            with open(args.test_list) as listfile:
                while True:
                    line = listfile.readline()
                    if not line:
                        break
                    data = line.split()

                    if len(data) == 2:
                        data = [random.randint(0, 1)] + data

                    files.append(Path(data[1]))
                    files.append(Path(data[2]))

            files = list(set(files))
        else:
            files = list(Path(args.test_list).glob('*/*.wav'))
        files.sort()

        same_smallest_score = 1
        diff_biggest_score = 0
        for f in tqdm(files):
            embed = speaker_model.embed_utterance(f,
                                          eval_frames=args.eval_frames,
                                          num_eval=num_eval,
                                          normalize=speaker_model.__L__.test_normalize)
            embed = embed.unsqueeze(-1)
            dist = F.pairwise_distance(embed, embeds).detach().cpu().numpy()
            dist = np.mean(dist, axis=0)
            score = 1 - np.min(dist) ** 2 / 2
            if classes[np.argmin(dist)] == f.parent.stem:
                if score < same_smallest_score:
                    same_smallest_score = score
                indexes = np.argsort(dist)[:2]
                if fabs((1 - dist[indexes[0]] ** 2 / 2) - (1 - dist[indexes[1]] ** 2 / 2)) < 0.001:
                    for i, idx in enumerate(indexes):
                        score = 1 - dist[idx] ** 2 / 2
                        if i == 0:
                            tqdm.write(f'+ {f}, {score} - {classes[idx]}',
                                       end='; ')
                        else:
                            tqdm.write(f'{score} - {classes[idx]}', end='; ')
                    tqdm.write('***')
                else:
                    tqdm.write(f'+ {f}, {score}', end='')
                    if score < args.test_threshold:
                        tqdm.write(' ***', end='')
                    tqdm.write('')
            else:
                if score > diff_biggest_score:
                    diff_biggest_score = score
                if score > args.test_threshold:
                    indexes = np.argsort(dist)[:3]
                    for i, idx in enumerate(indexes):
                        score = 1 - dist[idx] ** 2 / 2
                        if i == 0:
                            tqdm.write(f'- {f}, {score} - {classes[idx]}',
                                       end='; ')
                        else:
                            tqdm.write(f'{score} - {classes[idx]}', end='; ')
                    tqdm.write('***')
        print(f'same_smallest_score: {same_smallest_score}')
        print(f'diff_biggest_score: {diff_biggest_score}')
        sys.exit(1)


def evaluate_result(path="backup/Raw_ECAPA/result/private_test_results.txt",
                    ref="log_service/test_lst_truth.txt"):
    com = path
    assert os.path.isfile(ref) and os.path.isfile(com), f"Files not exists {ref} or {com}"

    ref_data = {}
    com_data = {}

    print("Checking results....")
    with open(ref, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=' ')
        for row in spamreader:
            key = f"{row[1]}/{row[-1]}"
            ref_data[key] = int(row[0])

    with open(com, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        for row in spamreader:
            key = f"{row[0]}/{row[1]}"
            com_data[key] = int(row[2])

    assert len(ref_data) == len(com_data), "The length of 2 files is not equal"
    assert list(ref_data.keys()) == list(
        com_data.keys()), "order is not matched"

    print("Test list infor:",
          f"Total '1' label: {list(ref_data.values()).count(1)} pairs",
          f"Total '0' label: {list(ref_data.values()).count(0)} pairs")

    # ROC
    fpr, tpr, thresholds = roc_curve(
        list(ref_data.values()), list(com_data.values()), pos_label=1)
    fnr = 1 - tpr
    tnr = 1 - fpr
    confussion_roc_matrix = "\nTPR: {:<24} | FNR: {:<24}\nFPR: {:<24} | TNR: {:<24}".format(
        tpr[1], fnr[1], fpr[1], tnr[1])
    accuracy = accuracy_score(list(ref_data.values()), list(com_data.values()))
    print("ROC evaluation:", confussion_roc_matrix, "\nAccuracy:", accuracy)

    # Precision-Recall
    beta_values = [0.5, 2]
    prec_recall = evaluate_by_precision_recall(
        list(ref_data.values()), list(com_data.values()), beta_values=[0.5, 2])
    print("Precision-Recall evaluation:\n", prec_recall[0])
    print("Accuracy for each class:",
          f"\n0's: {prec_recall[1][0]}\n1's: {prec_recall[1][1]}")
    for b in beta_values:
        print(f"f{b} score:", prec_recall[2][b])

    return confussion_roc_matrix, prec_recall[0]


def evaluate_by_precision_recall(y_true, y_pred, beta_values=[1]):
    target_names = ["Label '0'", "Label '1'"]
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
        fb_score = fbeta_score(y_true, y_pred, beta=b, pos_label=1)
        fb_scores[b] = fb_score

    return report, accuracy_per_classes, fb_scores


if __name__ == '__main__':
    pass
