import torch
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers import BertModel

tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-Japanese')

model = BertModel.from_pretrained('bert-base-japanese')

input_batch = \
    ["すもももももももものうち",
    "隣の客はよく柿食う客だ",
    "東京特許許可局許可局長"]

encoded_data = tokenizer.batch_encode_plus(
input_batch, pad_to_max_length=True, add_special_tokens=True)

print(encoded_data)
