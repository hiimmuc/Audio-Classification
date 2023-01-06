from curses.ascii import SP
import os
import csv
import importlib
import random
import itertools
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm

import onnx
import onnxruntime as onnxrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from processing.audio_loader import loadWAV
from utils import (similarity_measure, cprint)
from dataloader import test_data_loader, worker_init_fn


class WrappedModel(nn.Module):

    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU

    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)
    

class SpeakerEncoder(nn.Module):    
    def __init__(self, model, criterion, optimizer, features, classifier, device='cuda', gpu=0, ** kwargs) -> None:
        """Speaker encoder class

        Args:
            model (dict): model
            criterion (dict): loss function definition
            optimizer (_type_): _description_
            features (_type_): _description_
            device (_type_): _description_
            include_top (bool, optional): _description_. Defaults to False.
        """
        super(SpeakerEncoder, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.classifier = classifier
        
        self.gpu = gpu
        self.device = torch.device(f'{device}:{self.gpu}')
        
        self.n_mels = kwargs['n_mels']
        self.features = features.lower()       
        
        if self.features in ['mfcc', 'melspectrogram']:
            Features_extractor = importlib.import_module(
                'models.encoder.FeatureExtraction.feature').__getattribute__(f"{features.lower()}")
            self.compute_features = Features_extractor(**kwargs).to(self.device)
        else:
            self.compute_features = None   
        
        SpeakerNetModel = importlib.import_module(
            'models.encoder.' + self.model['encoder']).__getattribute__('MainModel')
        self.__E__ = SpeakerNetModel(nOut=self.model['nOut'], 
                                     features = self.features, 
                                     device=self.device, 
                                     model = self.model, **kwargs).to(self.device)
        
        SpeakerClassification = importlib.import_module(
            'models.decoder.decoder').__getattribute__(self.model['decoder'])
        self.__D__ = SpeakerClassification(
            in_feats=self.classifier['input_size'],
            out_feats=self.classifier['out_neurons'])

        ###
        LossFunction = importlib.import_module(
            'losses.' + self.criterion['name']).__getattribute__(f"{self.criterion['name']}")

        self.__L__ = LossFunction(nOut=self.model['nOut'], 
                                  margin=self.criterion['margin'], 
                                  scale=self.criterion['scale'],
                                  **kwargs)
        #NOTE: nClasses is defined
        try:
            self.test_normalize = self.__L__.test_normalize
        except AttributeError:
            # not include in the loss init params
            self.test_normalize = False
        
        self.nPerSpeaker = kwargs['dataloader_options']['nPerSpeaker']
        
    def forward(self, data, label=None):
        # data size: n_speaker x bsize x n_samples
        data = data.reshape(-1, data.size()[-1]).unsqueeze(0).to(self.device) # -> 1 x (n_speaker x batch_size) x n_samples
        # data = data.to(self.device)
        feat = []
        output_cls = []
        # forward n utterances per speaker and stack the output
        for inp in data:
            if self.features != 'raw':
                inp = self.compute_features(inp) # convert raw audio to mel
                
            emb = self.__E__.forward(inp.to(self.device))
            
            ## Beta for the recognition task
            output = self.__D__.forward(emb.to(self.device))

            #              
            feat.append(emb)
            output_cls.append(output)

        feat = torch.stack(feat, dim=1).squeeze()
        output_cls = torch.stack(output_cls, dim=1).squeeze()

        ## Calculate loss
        if label == None:
            return output_cls
        else:
            output_cls = output_cls.reshape(
                self.nPerSpeaker, -1, output_cls.size()[-1]).transpose(1, 0).squeeze(1)
            # output_cls: batch size x n_n_classes
            nloss, prec1 = self.__L__.forward(
                output_cls, label.to(self.device))
            return nloss, prec1
                

class ModelHandling(object):
    def __init__(self, encoder_model,
                 optimizer='adam', 
                 callbacks='steplr',  
                 device='cuda',
                 gpu=0, mixedprec=False, **kwargs):
        """_summary_

        Args:
            encoder_model (_type_): _description_
            optimizer (str, optional): _description_. Defaults to 'adam'.
            callbacks (str, optional): _description_. Defaults to 'steplr'.
            device (str, optional): _description_. Defaults to 'cuda'.
            gpu (int, optional): _description_. Defaults to 0.
            mixedprec (bool, optional): _description_. Defaults to False.
        """
        
        # take only args
        super(ModelHandling, self).__init__()
        
        self.kwargs = kwargs
        
        self.save_path = self.kwargs['save_folder']
        self.audio_spec = self.kwargs['audio_spec']
        
        self.T_max = 0 if 'T_max' not in self.kwargs else self.kwargs['T_max']

        self.__model__ = encoder_model
        self.model_name = self.__model__.module.model['encoder']
        self.criterion = self.__model__.module.criterion['name']
        
        self.scaler = GradScaler()

        self.gpu = gpu
        self.device = torch.device(f"{device}:{gpu}")
        
        self.augment = kwargs['augment']
        self.augment_chain = kwargs['augment_options']['augment_chain']            
        self.mixedprec = mixedprec
        
        Optimizer = importlib.import_module(
            'optimizer.' + optimizer['name']).__getattribute__(f"{optimizer['name']}")
        self.__optimizer__ = Optimizer(self.__model__.parameters(),
                                       weight_decay=optimizer['weight_decay'],
                                       lr_decay=optimizer['lr_decay'],
                                       **kwargs)

        self.callback = callbacks

        if self.callback['name'] in ['steplr', 'cosine_annealinglr_pt', 'cycliclr']:
            Scheduler = importlib.import_module(
                'callbacks.torch_callbacks').__getattribute__(f"{self.callback['name'].lower()}")
            self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__,                                                          
                                                         lr_decay=optimizer['lr_decay'], **dict(kwargs, T_max=self.T_max))
        elif self.callback['name'] == 'reduceOnPlateau':
            Scheduler = importlib.import_module(
                'callbacks.' + callbacks).__getattribute__('LRScheduler')
            self.__scheduler__ = Scheduler(self.__optimizer__, 
                                           step_size=self.callback['step_size'], 
                                           lr_decay=optimizer['lr_decay'], 
                                           patience=self.callback['step_size'], 
                                           min_lr=self.callback['base_lr'], factor=0.95)
            self.lr_step = 'epoch'
        else:
            raise 'Invalid callbacks'
        
        assert self.lr_step in ['epoch', 'iteration']
        
                ####Print information only on main process
        
        if self.gpu == 0:
            nb_params = sum([param.view(-1).size()[0]
                            for param in self.__model__.module.__E__.parameters() if param.requires_grad])
            print("Model information:")
            print(f"- Initialized model: {self.__model__.module.model['encoder']} - Trainable params: {nb_params:,}")
            print(f"- Using loss function: {self.__model__.module.criterion['name']}")
            print(f"- Embedding normalized: ", self.__model__.module.test_normalize)
            print(f"- Using callback: {self.callback['name']}")
            print(f"- Using optimizer: {optimizer['name']}")           
        # ================================================================================================

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====
    def fit(self, loader, epoch=0, verbose=True):
        '''Train

        Args:
            loader (Dataloader): dataloader of training data
            epoch (int, optional): [description]. Defaults to 0.

        Returns:
            tuple: loss and precision
        '''
        self.__model__.train()

        counter = 0
        loss = 0
        top1 = 0  # EER or accuracy
           
        if verbose:
            loader_bar = tqdm(loader, desc=f">EPOCH_{epoch}", unit="it", colour="green", total=self.T_max)
        else:
            loader_bar = loader

        for (data, data_label) in loader_bar:
            # if self.augment and 'time_domain_torch' in self.augment_chain:
            #     data = self.augment_on-stream_engine.forward(data)
                
            data = data.transpose(0, 1) # batch, channel, num_samples -> channel, batch, num_samples
            
            self.__model__.zero_grad()
            
            label = torch.LongTensor(data_label)
            
            if self.mixedprec:
                with autocast():
                    nloss, prec1 = self.__model__.forward(data, label)
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss, prec1 = self.__model__.forward(data, label)
                nloss.backward()
                self.__optimizer__.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1
            counter += 1

            # update tqdm bar
            if verbose:
                loader_bar.set_postfix(LR=f"{round(float(self.__optimizer__.param_groups[0]['lr']), 8)}", 
                                       TLoss=f"{round(float(loss / counter), 5)}", 
                                       TAcc=f"{round(float(top1 / counter), 3)}%")

            if self.lr_step == 'iteration':
                self.__scheduler__.step()

        # select mode for callbacks
        if self.lr_step == 'epoch' and self.callback not in ['reduceOnPlateau', 'auto']:
            self.__scheduler__.step()

        elif self.callback == 'reduceOnPlateau':
            # reduce on plateau
            self.__scheduler__(loss / counter)

        elif self.callback == 'auto':
            if epoch <= 50:
                self.__scheduler__['rop'](loss / counter)
            else:
                if epoch == 51:
                    if verbose:
                        cprint("\n[INFO] # Epochs > 50, switch to steplr callback\n========>\n", 'r')
                self.__scheduler__['steplr'].step()

        loss_result = loss / (counter)
        precision = top1 / (counter)

        return loss_result, precision
    
    def predict(self, inp):
        return self.__model__.forward(inp)
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate model with eval files
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def evaluateFromList(self,
                         listfilename,
                         distributed,
                         dataloader_options,
                         cohorts_path='checkpoint/dump_cohorts.npy',
                         num_eval=10,
                         scoring_mode='cosine', **kwargs):
        """_summary_

        Args:
            listfilename (_type_): _description_
            distributed (_type_): _description_
            dataloader_options (_type_): _description_
            cohorts_path (str, optional): _description_. Defaults to 'checkpoint/dump_cohorts.npy'.
            num_eval (int, optional): _description_. Defaults to 10.
            scoring_mode (str, optional): _description_. Defaults to 'cosine'.

        Returns:
            _type_: _description_
        """
        
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}

        # Cohorts
        cohorts = None
        if cohorts_path is not None and scoring_mode == 'norm':
            cohorts = np.load(cohorts_path)

        # Read all lines
        with open(listfilename) as listfile:
            lines = listfile.readlines()
                            
        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        test_dataset = test_data_loader(test_list=setfiles,
                                        audio_spec=self.audio_spec, 
                                        num_eval=num_eval, **kwargs)
        
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, shuffle=False)
        else:
            sampler = None
            
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=dataloader_options['num_workers'],
            worker_init_fn=None,
            pin_memory=True,
            drop_last=False,
            sampler=sampler
        )
        
        print(">>>>Evaluation")

        # Save all features to dictionary
        loader_bar = tqdm(
            test_loader, desc=">>>>Reading file: ", unit="files", colour="red")
        ##
        for idx, src in enumerate(loader_bar):
            
            audio, filename = src
            inp1 = torch.FloatTensor(audio).to(self.device)  
            
            with torch.no_grad():
                ref_feat = self.__model__.forward(inp1).detach().cpu()
                                
            feats[str(Path(str(filename[0])))] = ref_feat       
        #
        all_scores = []
        all_labels = []
        all_trials = []
        
        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)
        
        if rank == 0:
            # run on main worker
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)
            
            # Read files and compute all scores

            for idx, line in enumerate(tqdm(lines, desc=">>>>Computing files", unit="pairs", colour="MAGENTA")):
                data = line.split()
                #label, audio1, audio2
                ref_feat = feats[data[1]].to(self.device)
                com_feat = feats[data[2]].to(self.device)

                if self.__model__.module.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                # NOTE: distance(cohort = None) for training, normalized score for evaluating and testing
                if cohorts_path is None:
                    dist = F.pairwise_distance(
                        ref_feat.unsqueeze(-1),
                        com_feat.unsqueeze(-1).transpose(
                            0, 2)).detach().cpu().numpy()
                    score = -1 * np.mean(dist)
                else:
                    if scoring_mode == 'norm':
                        score = similarity_measure('zt_norm',
                                                ref_feat,
                                                com_feat,                                                
                                                cohorts,
                                                top=200)
                    elif scoring_mode == 'cosine':
                        score = similarity_measure('cosine',ref_feat, com_feat)
                    elif scoring_mode == 'pnorm' :
                        score = similarity_measure('pnorm', ref_feat, com_feat, p = 2)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])             
            
        return all_scores, all_labels, all_trials
    
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Testing files from list of testing pairs
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def testFromList(self,
                     test_list='verification_file.txt',
                     thresh_score=0.5,
                     distributed=False,
                     dataloader_options=None,
                     cohorts_path=None,
                     num_eval=10,
                     scoring_mode='norm',
                     output_file=None):
        """_summary_

        Args:
            root (_type_): _description_
            test_list (str, optional): _description_. Defaults to 'evaluation_test.txt'.
            thresh_score (float, optional): _description_. Defaults to 0.5.
            cohorts_path (_type_, optional): _description_. Defaults to None.
            num_eval (int, optional): _description_. Defaults to 10.
            scoring_mode (str, optional): _description_. Defaults to 'norm'.
            output_file (_type_, optional): _description_. Defaults to None.
        """
        self.__model__.eval()

        lines = []
        files = []
        feats = {}

        # Cohorts
        cohorts = None
        if cohorts_path is not None and scoring_mode == 'norm':
            cohorts = np.load(cohorts_path)
        save_root = os.path.join(self.save_path , f"{self.__model__.module.model['encoder']}/{self.__model__.module.criterion['name']}/result")

        
        read_file = Path(test_list)
        if output_file is None:
            output_file = test_list.replace('.txt','_result.txt')
        write_file = Path(save_root, output_file) if os.path.split(output_file)[0] == '' else output_file # add parent dir if not provided
        
        # Read all lines from testfile (read_file)
        print(">>>>TESTING...")
        print(f">>> Threshold: {thresh_score}")
        with open(read_file, newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in spamreader:
                files.append(row[0])
                files.append(row[1])
                lines.append(row)

        setfiles = list(set(files))
        setfiles.sort()

        # Save all features to feat dictionary
        for idx, filename in enumerate(tqdm(setfiles, desc=">>>>Reading file: ", unit="files", colour="red")):
            audio = loadWAV(filename, 
                            self.audio_spec,
                            evalmode=True,
                            augment=False,
                            augment_options=[],
                            num_eval=num_eval,
                            random_chunk=False)
            
            inp1 = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

            with torch.no_grad():
                ref_feat = self.__model__.forward(inp1).detach().cpu()
                
            feats[filename] = ref_feat

        # Read files and compute all scores
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['audio_1', 'audio_2', 'pred_label' , 'score'])
            for idx, data in enumerate(tqdm(lines, desc=">>>>Computing files", unit="pairs", colour="MAGENTA")):
                ref_feat = feats[data[0]].to(self.device)
                com_feat = feats[data[1]].to(self.device)

                if self.__model__.module.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                if cohorts_path is None:
                    dist = F.pairwise_distance(
                        ref_feat.unsqueeze(-1),
                        com_feat.unsqueeze(-1).transpose(
                            0, 2)).detach().cpu().numpy()
                    score = -1 * np.mean(dist)
                else:
                    if scoring_mode == 'norm':
                        score = similarity_measure('zt_norm',ref_feat,
                                                    com_feat,
                                                    cohorts,
                                                    top=200)
                    elif scoring_mode == 'cosine':
                        score = similarity_measure('cosine',ref_feat, com_feat)
                    elif scoring_mode == 'pnorm' :
                        score = similarity_measure('pnorm', ref_feat, com_feat, p = 2)

                pred = '1' if score >= thresh_score else '0'
                spamwriter.writerow([data[0], data[1], pred, score])
   
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## preparing the cohort or embeddings of multiple utterances
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def prepare(self,
                save_path=None,
                prepare_type='cohorts',
                num_eval=10,
                source=None, **kwargs):     
                       
        """ Prepared 1 of the 2:
        1. Mean L2-normalized embeddings for known speakers.
        2. Cohorts for score normalization.
        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        
        self.__model__.eval()
        if not source:
            raise "Please provide appropriate source!"
        ########### cohort preparation
        if prepare_type == 'cohorts':
            n_emb_per_spk = 3

            assert isinstance(source, str), "Please provide path to train metadata files"
            read_file = Path(source)

            lines = []
            cohort_spk_files = dict()
            cohort_embedding = dict()

            with open(read_file, 'r') as listfile:
                lines = [line.replace('\n', '') for line in listfile.readlines()]
                
            for line in tqdm(lines, desc='Gathering files...', unit=' files'):
                data = line.split()
                spkID, path = data[:2]
                cohort_spk_files.setdefault(spkID, []).append(path)
                
            for spkID, paths in tqdm(cohort_spk_files.items(), unit=' speakers', desc='Getting speaker embedding'):
                for path in paths[:n_emb_per_spk]:
                    emb = self.embed_utterance(path, num_eval=num_eval, normalize=True)
                    cohort_embedding.setdefault(spkID, []).append(emb)

            cohort_speakers = list(cohort_embedding.keys())
            cohort = np.vstack([np.mean(np.vstack(cohort_embedding[speaker]), axis=0, keepdims=True) 
                                for speaker in cohort_speakers])
            if save_path:
                np.save(save_path, np.array(cohort))
            return True
                
        ############# Embedding preparation
        elif prepare_type == 'embed':
            # Prepare mean L2-normalized embeddings for known speakers.
            # load audio from_path (root path)
            # option 1: from root
            if isinstance(source, str):
                speaker_dirs = [x for x in Path(source).iterdir() if x.is_dir()]
                embeds = None
                classes = {}
                # Save mean features
                for idx, speaker_dir in enumerate(speaker_dirs):
                    classes[idx] = speaker_dir.stem
                    files = list(speaker_dir.glob('*.wav'))
                    mean_embed = None
                    embed = None
                    for f in files:
                        embed = self.embed_utterance(
                            f,
                            num_eval=num_eval,
                            normalize=self.__model__.module.test_normalize)
                        if mean_embed is None:
                            mean_embed = embed.unsqueeze(0)
                        else:
                            mean_embed = torch.cat(
                                (mean_embed, embed.unsqueeze(0)), 0)
                    mean_embed = torch.mean(mean_embed, dim=0)
                    if embeds is None:
                        embeds = mean_embed.unsqueeze(-1)
                    else:
                        embeds = torch.cat((embeds, mean_embed.unsqueeze(-1)), -1)

                # print(embeds.shape)
                # embeds = rearrange(embeds, 'n_class n_sam feat -> n_sam feat n_class')
                if save_path:
                    torch.save(embeds, Path(save_path, 'embeds.pt'))
                    np.save(str(Path(save_path, 'classes.npy')), classes)
                return True
                
            elif isinstance(source, list):
                #option 2: list of audio in numpy format
                mean_embed = None
                embed = None
                for audio_data_np in source:
                    embed = self.embed_utterance(audio_data_np,  
                                                 num_eval=num_eval,
                                                 normalize=self.__model__.module.test_normalize)
                    if mean_embed is None:
                        mean_embed = embed.unsqueeze(0)
                    else:
                        mean_embed = torch.cat(
                            (mean_embed, embed.unsqueeze(0)), 0)
                mean_embed = torch.mean(mean_embed, dim=0)
                
                if save_path:
                    torch.save(mean_embed, Path(save_path, 'embeds.pt'))
                return mean_embed                
        else:
            raise NotImplementedError
        
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## get embedding of a single utterance
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def embed_utterance(self,
                        source,
                        num_eval=20,
                        normalize=False):
        """_summary_

        Args:
            source (_type_): _description_
            num_eval (int, optional): _description_. Defaults to 10.
            normalize (bool, optional): _description_. Defaults to False.
 
        Returns:
            _type_: _description_
        """

        audio = loadWAV(source,
                        self.audio_spec,
                        evalmode=True,
                        augment=False,
                        augment_options=[],
                        num_eval=num_eval,
                        random_chunk=False)

        inp = torch.FloatTensor(audio).to(self.device)
                    
        with torch.no_grad():
            embed = self.__model__.forward(inp).detach().cpu()
        if normalize:
            embed = F.normalize(embed, p=2, dim=1)
        return embed

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        try:
            state_dict = self.__model__.module.state_dict()
        except:
            state_dict = self.__model__.state_dict()

        torch.save(state_dict, path)

    def loadParameters(self, path, show_error=True):
        if os.path.exists(path):
            try:
                self_state = self.__model__.module.state_dict()
            except:
                self_state = self.__model__.state_dict()
            
            loaded_state = torch.load(path, map_location=self.device)

            for name, param in loaded_state.items():
                origname = name

                if name not in self_state:
                    name = name.replace("module.", "")

                    if name not in self_state:
                        if show_error:
                            print("{} is not in the model.".format(origname))
                        continue

                if self_state[name].size() != loaded_state[origname].size():
                    if show_error:
                        print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                            origname, self_state[name].size(), loaded_state[origname].size()))
                    continue

                self_state[name].copy_(param)
        else:
            raise "Model's path is not exists"

    def export_onnx(self, state_path, check=True):
        save_root = self.save_path + f"/{self.model_name}/{self.criterion}/model"
        save_path = os.path.join(save_root, f"model_eval_{self.model_name}.onnx")

        input_names = ["input"]
        output_names = ["output"]
        # NOTE: because using torch.audio -> cant export onnx
        # nume_val, samplerate * (1 + win_len - hoplen)
        n_s = self.kwargs['num_eval']
        dim = int(self.audio_spec['sample_rate'] * self.audio_spec['sentence_len'])
        dummy_input = torch.randn(n_s, dim, device="cuda")

        self.loadParameters(state_path)
        self.__model__.eval()

        torch.onnx.export(self.__model__,
                          dummy_input,
                          save_path,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names,
                          export_params=True,
                          opset_version=11)

        # double check
        if os.path.exists(save_path) and check:
            print("checking export")
            model = onnx.load(save_path)
            onnx.checker.check_model(model)
            print(onnx.helper.printable_graph(model.graph))
            cprint("Done!!!", 'r')

    def onnx_inference(self, model_path, inp):
        def to_numpy(tensor):
            if not torch.is_tensor(tensor):
                tensor = torch.FloatTensor(tensor)
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
        onnx_session = onnxrt.InferenceSession(model_path)
        onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(inp)}
        onnx_output = onnx_session.run(None, onnx_inputs)
        return onnx_output
##################################################################################################