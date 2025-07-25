import datasets
from datasets import load_dataset


ds = load_dataset("wikitext", "wikitext-2-raw-v1")   # ds is {'train', 'validation', 'test'}

train_split = ds["train"]            # a `datasets.Dataset` object
#print(len(train_split))              # number of rows
#print(train_split.features)          # column schema → only "text"

# --- look at actual rows -----------------------------------------
print(train_split[0])                # {'text': ' David Livingstone , ...'}
print(train_split[1]['text'])                # {'text': ' David Livingstone , ...'}
print(train_split[2])                # {'text': ' David Livingstone , ...'}
print(train_split[3])                # {'text': ' David Livingstone , ...'}
print(train_split[4])                # {'text': ' David Livingstone , ...'}
print(train_split[5])                # {'text': ' David Livingstone , ...'}
print(train_split[6])                # {'text': ' David Livingstone , ...'}
print(train_split[7])                # {'text': ' David Livingstone , ...'}
print(train_split[8])                # {'text': ' David Livingstone , ...'}
print(train_split[9])                # {'text': ' David Livingstone , ...'}
print(train_split[10])                # {'text': ' David Livingstone , ...'}
print(train_split[11])                # {'text': ' David Livingstone , ...'}
print(train_split[12])                # {'text': ' David Livingstone , ...'}
print(train_split[13])                # {'text': ' David Livingstone , ...'}
print(train_split[14])                # {'text': ' David Livingstone , ...'}
print(train_split[15])                # {'text': ' David Livingstone , ...'}

#print(train_split[:5])               # first five rows (list of dicts)
