# NEED TO RUN THESE TWO COMMANDS IN TERMINAL BEFORE RUNNING THIS SCRIPT
# git lfs install
# git clone https://huggingface.co/datasets/xsum
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import numpy as np
from typing import List, Union
import nltk
import pandas as pd
     

from datasets import load_dataset, Dataset, DatasetDict
import glob
import os
DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"

NMT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-mul"
NMT_MODEL_NAME = "jkorsvik/opus-mt-eng-nor"
NMT_MODEL_CONFIG = {}

TRANSLATION_PREFIX = ">>nob<< "
PADDING_VALUE = 54775
PADDING_TOKEN = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKOWN_TOKEN = "<unk>"

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

MAX_INPUT_LENGTH = 512
BATCH_SIZE = 32
OUTPUT_DIR = "translated_data"


class Opus:
    def __init__(self,model_name: str, device:str = None):
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.max_length = None
    
    def translate(self, documents: Union[str, List[str]], show_progress_bar: bool = False, beam_size: int = 5, 
                        batch_size: int = 16, paragraph_split: str = "\n"):

        if isinstance(documents, str):
            translated_sent = self.model_translator(documents)
            return translated_sent[0]
        else:
            # Split document into sentences
            splitted_sentences = []
            sent2doc = []
            for doc in documents:
                paragraphs = doc.split(paragraph_split) if paragraph_split is not None else [doc]
                for para in paragraphs:
                    for sent in self._sentence_splitting(para.strip()):
                        sent = sent.strip()
                        if len(sent) > 0:
                            splitted_sentences.append(sent)
                sent2doc.append(len(splitted_sentences))

                translated_sentences = self.translate_sentences(splitted_sentences, show_progress_bar=show_progress_bar, beam_size=beam_size, batch_size=batch_size)

            # Merge sentences back to documents
            translated_docs = []
            for doc_idx in range(len(documents)):
                start_idx = sent2doc[doc_idx - 1] if doc_idx > 0 else 0
                end_idx = sent2doc[doc_idx]
                translated_docs.append(self._reconstruct_document(documents[doc_idx], splitted_sentences[start_idx:end_idx], translated_sentences[start_idx:end_idx]))

            return translated_docs

    def translate_sentences(self, sentences: Union[str, List[str]], show_progress_bar: bool = False, beam_size: int = 5, batch_size: int = 32):
        #Sort by length to speed up processing
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        iterator = range(0, len(sentences_sorted), batch_size)
        output = []

        if show_progress_bar:
            scale = min(batch_size, len(sentences))
            iterator = tqdm.tqdm(iterator, total=len(sentences)/scale, unit_scale=scale, smoothing=0)

        for start_idx in iterator:
            output.extend(self.model_translator(sentences_sorted[start_idx:start_idx+batch_size], beam_size=beam_size))

        #Restore original sorting of sentences
        output = [output[idx] for idx in np.argsort(length_sorted_idx)]
        
        return output

    def model_translator(self, sentences:str, beam_size:int=5):
        self.model.to(self.device)

        inputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt").input_ids
        inputs = inputs.to(self.device)

        with torch.no_grad():
            translated = self.model.generate(inputs, num_beams=beam_size)
            output = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return output

    @staticmethod
    def _sentence_splitting(text: str):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        return nltk.sent_tokenize(text)

    @staticmethod
    def _reconstruct_document(doc, org_sent, translated_sent):
          sent_idx = 0
          char_idx = 0
          translated_doc = ""
          while char_idx < len(doc):
              if sent_idx < len(org_sent) and doc[char_idx] == org_sent[sent_idx][0]:
                  translated_doc += translated_sent[sent_idx]
                  char_idx += len(org_sent[sent_idx])
                  sent_idx += 1
              else:
                  translated_doc += doc[char_idx]
                  char_idx += 1
          return translated_doc
    
class NMT:
    def __init__(self, model:str, dataset_name, file_id, data_range_to_translate , data_type, skip_cols_to_translate, file_folder_drive ):
        self.opus = model
        self.dataset_name= dataset_name
        self.file_id = file_id
        self.data_range_to_translate = data_range_to_translate
        self.data_type = data_type
        self.skip_cols_to_translate = skip_cols_to_translate
        self.file_folder_drive = file_folder_drive
        self.dataset = load_dataset(self.dataset_name, CONFIG, split=f'{self.data_type}{self.data_range_to_translate}')
        self.path = f'./{self.file_folder_drive}/df_{self.dataset_name}_{self.data_type}{str(self.file_id)}.csv'

        # print(self.dataset)

    def translate_dataset(self, batch_size:int = 64, beam_size:int=5):
        df_data_temp = []
        col_list_dataset = list(self.dataset[0].keys())

        for row_data in tqdm(self.dataset,desc='Translation in progress') :
            row_data_temp = []
            for c_list in col_list_dataset:
                to_translate_row =row_data[c_list]
                if c_list in skip_cols_to_translate:
                    is_translated_row = to_translate_row
                else:
                    is_translated_row = self.opus.translate(to_translate_row, batch_size=batch_size,beam_size=beam_size)
                row_data_temp.append(is_translated_row)
            data_zip = tuple(row_data_temp)
            df_data_temp.append(data_zip)

        return pd.DataFrame(df_data_temp, columns = col_list_dataset)

    def subdata_to_drive(self, df_trans):
      with open(self.path, 'w', encoding = 'utf-8') as f:
        df_trans.to_csv(f, sep=";")

    @staticmethod
    def assemble_df_data_type(file_folder_drive,dataset_name,data_type):
      # setting the path for joining multiple files
      files = os.path.join(f'./{file_folder_drive}/', f'df_{dataset_name}_{data_type}*.csv')
      files = glob.glob(files)
      files.sort()
      df_list = [pd.read_csv(f, sep=';') for f in files]  
      df = pd.concat(df_list, ignore_index=True)    
      df_merged = df.copy()
      df_merged = df_merged.iloc[: , 1:]
      return Dataset.from_pandas(df_merged)



if __name__ == "__main__":

    model = Opus(NMT_MODEL_NAME)


    indexes = [0,13000, 23000, 33000, 45000]

    dataset_name = "cnn_dailymail"
    file_id=1
    data_type = "train"
    file_folder_drive = "cnn_dailymail"
    data_range_to_translate= f"[{indexes[file_id-1]}:{indexes[file_id]}]"
    skip_cols_to_translate=['id']
    import time
    #time.sleep(60*60*5)
    nmt = NMT( model, dataset_name, file_id, data_range_to_translate , data_type, skip_cols_to_translate, file_folder_drive)


    df_trans= nmt.translate_dataset(batch_size=64)
    nmt.subdata_to_drive(df_trans)



    #train_dataset = nmt.assemble_df_data_type("xsum", "xsum","train")
    #train_dataset = nmt.assemble_df_data_type("xsum", "xsum","train")
    #test_dataset = nmt.assemble_df_data_type("xsum", "xsum","test")
    # val_dataset =  nmt.assemble_df_data_type("xsum", "xsum","val")



    #master_dataset_dict = DatasetDict({"test":test_dataset})#, "train":train_dataset,"validation":val_dataset})
