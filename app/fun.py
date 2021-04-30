import pickle
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
import torch.nn as nn
# from reformer_pytorch import ReformerLM,Autopadder
# from performer_pytorch import PerformerLM

# https://pytorch-crf.readthedocs.io/en/stable/
from torchcrf import CRF  

# from reformer_pytorch.generative_tools import TrainingWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
# from performer_pytorch import PerformerLM
from transformers import BertTokenizer, BertModel,BertForMaskedLM,AutoModelForMaskedLM,AutoModelForTokenClassification,ElectraPreTrainedModel,ElectraModel
import re
from argparse import ArgumentParser
# hparams=parse_args()
from transformers import AutoTokenizer,AutoModel,BertForMaskedLM,AutoModelForMaskedLM,BertModel,BertTokenizer
# https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
from pytorch_lightning.metrics import functional as FM
from torch.nn import CrossEntropyLoss, MSELoss
# from omegaconf import OmegaConf



class LitAutoMark(pl.LightningModule):
    """NER类似标注
    参考 https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForSequenceClassification
    
    https://github.com/liuyukid/transformers-ner/blob/f3bfcad46526c9cc454e3fdc49c6d19da5c419b7/models/electra_ner.py#L70
    """

    def __init__(self,max_seq_len=4096,learning_rate=1e-4,warmup=1000,num_labels=2,frequency=1,patience=100,T_max=100,**kwargs):
        """
        
        https://pytorch-lightning.readthedocs.io/en/1.2.4/common/hyperparameters.html
        """
        super().__init__()
#         print(**kwargs)
#         将参数存进hparams
#         conf = OmegaConf.create({"num_labels" : 2,**kwargs})
#         self.save_hyperparameters(conf)
        
        
        self.num_labels=num_labels
#         conf = OmegaConf.create(...)
#         self.hparams = hparams
#         self.save_hyperparameters()

#         self.model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext") 
        self.model = AutoModel.from_pretrained("hfl/chinese-electra-180g-small-ex-discriminator")
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-electra-180g-small-ex-discriminator')
        self.classifier = torch.nn.Linear(in_features=256, out_features=self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        #         self.funct=torch.nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.labels=["X","B-w","I-w","E-w"]
# labels
#         self.init_weights()
#     def from_pretrained(self,file='../input/reformertrainfrombertmodel/ReformerLM_model_state_dict.bin'):
#         """加载预训练模型"""
# #         from_pretrained
#         self.model.load_state_dict(torch.load(file))
#         self.model.train()
#         pass
        # del self.bert
    def forward(self, x,y=None,attention_mask=None,return_pred=True):
        """训练梯度"""
        # in lightning, forward defines the prediction/inference actions

        out = self.model(x,attention_mask=attention_mask)
#         print(out.keys())
        last_hidden_state=out.last_hidden_state
#         print("last_hidden_state",last_hidden_state.size())
        last_hidden_state = self.dropout(last_hidden_state) # 获取第一个输出

        logits=self.classifier(last_hidden_state)
        #调整矩阵形状.permute(1,0)
        loss=None
        pred=None
        
        if attention_mask!=None:
            attention_mask=attention_mask.byte()
#         print(attention_mask.size(),last_hidden_state.size())
        if y!=None:
    #         print(last_hidden_state.size(),tags.size())
            loss=-1 * self.crf(emissions=logits,tags=y,mask=attention_mask)# reduction="token_mean",
        
        if return_pred:
#             pred=torch.tensor(self.crf.decode(last_hidden_state,mask=attention_mask)).to(self.device)
            pred=torch.tensor(self.crf.decode(emissions=logits)).to(self.device)
        if return_pred:
        
            return pred,loss
        else:
            return loss
    def is_contains_chinese(self,strs):
        """检测是不是包含中文字符"""
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False
    def decode(self,text):
#         datas="""
#         冠心病；频发房早；高血压；肾功不全 白内

#         """
        max_length=128
        if not self.is_contains_chinese(text) or len(text)>max_length:
            return [],[],[text]
        # datas=datas.replace("；","[PAD]")
        labels=["X","B-w","I-w","E-w"]
        newData=[]
        
        textLen=len(text)
        faketext=re.split("。|。|！|？|？|\r|\t|；|;|,|，|,|、|，| |\n",text)

        
        new=" ".join(faketext)
        
#         print(new)
        for w in new:
#             w=list(w)
            if w==" ":
                w="[PAD]"
            
            
#             w=w.replace(" ","[PAD]")
            
            newData.append(w)
        
        wds=" ".join(newData)
        
#         max_length=128
#         print(wds)
        o_wds=wds.split(" ")
        tokdatas=self.tokenizer(wds,return_tensors="pt",max_length=max_length,truncation=True)
#     tokdatas=self.tokenizer(wds,return_tensors="pt", padding="max_length",max_length=max_length,truncation=True)
        perd,_=self(tokdatas['input_ids'],attention_mask=tokdatas['attention_mask'],return_pred=True)
#         perd
        words=[]
        data=[]
        for item_ids in perd.tolist():
#             print(wds,item_ids)
            word=[]
            data=[]
            dataw=[]
#             wds=wds.replace("[PAD]"," ")
#             print("o_wds",o_wds)
            end=True
#     text
            for l,w in zip(item_ids[1:-1],o_wds):
#                 print(w,labels[l])
                data.append((w,labels[l]))
#                 if w in ["[CLS]","[SEP]"]:
#                     continue
                if w=="[PAD]":
                    w=" "
                if labels[l]=="B-w":
                    word=[]
                    if end==True:
                        dataw.append("-;-"+w)
                        end=False
                    else:
                        dataw.append(w)
                        end=False
                    word.append(w)
                elif labels[l]=="I-w":
                    word.append(w)
                    dataw.append(w)
                elif labels[l]=="E-w":
                    word.append(w)
                    dataw.append(w)
                    words.append(word)
                    word=[]
                    end=True
                else:
                    if end==True:
                        dataw.append("-;-"+w)
                        end=False
                    else:
                        dataw.append(w)
                        
                    word.append(w)
                    
#             print(dataw)
#             print("".join(dataw))
                dw="".join(dataw)
                out=dw.split("-;-")
                new=[]
                for w in out:
                    if w not in [" ","","\t","\n"]:
                        new.append(w.strip())
            return words,data,new
#         tokdatas
model=LitAutoMark(num_labels=4)
model.load_state_dict(torch.load("./model/LitAutoMark.bin",map_location=torch.device('cpu')))
model.eval()
# print("ss")     


# for text in data[:20]:
#     print("\n\n")
#     wd=model.decode(text[1])
#     words=[]
#     words_t=[]
#     print(text[1])
#     print(wd[-1])
#     for w in wd:
#         print(w)
#         words.append("".join(wd))
#         words_t.append(w[1])
#     print(text[0].replace("\n",""),"\n",words,words_t,"\n\n\n")