import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset # type: ignore
from Tokenizer import get_or_build_tokenizer


class BilingualDataset(nn.Module):

  def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
    super().__init__()

    self.dataset = dataset
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt

    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.seq_len = seq_len

    self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index: any):
    src_target_pair = self.dataset[index]
    src_text = src_target_pair['translation'][self.src_lang]
    tgt_txt = src_target_pair['translation'][self.tgt_lang]

    encoder_input_tokens = self.tokenizer_src.encode(src_text).ids[:self.seq_len - 2]  # Truncate
    decoder_input_tokens = self.tokenizer_tgt.encode(tgt_txt).ids[:self.seq_len - 1]  # Truncate

    encoder_num_padding_tokens = self.seq_len - len(encoder_input_tokens) - 2
    decoder_num_padding_tokens = self.seq_len - len(decoder_input_tokens) - 1

    if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
      raise ValueError("Sentence is Too Long")

    # Ensure padding values are non-negative
    encoder_num_padding_tokens = max(0, encoder_num_padding_tokens)
    decoder_num_padding_tokens = max(0, decoder_num_padding_tokens)

    # Adds SOS and EOS tokens to the Encoder Input
    encoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(encoder_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
        ]
    )

    # Adds SOS tokens to the Decoder Input
    decoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(decoder_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
        ]
    )

    # Add EOS tokens to the label (What We expect as an output from the decoder)
    label = torch.cat(
        [
            torch.tensor(decoder_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
        ]
    )

    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return {
        "encoder_input": encoder_input, # (seq_len)
        "decoder_input": decoder_input, # (seq_len)
        "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
        "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len),
        "label": label, # (seq_len)
        "src_text": src_text,
        "tgt_text": tgt_txt
    }

def causal_mask(size):
  mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
  return mask == 0

    
def get_dataset(config):
  dataset_raw = load_dataset('Helsinki-NLP/opus_books', data_dir=f"{config['src_lang']}-{config['tgt_lang']}", split='train[:20%]')

  # Build Tokenizers
  tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['src_lang'])
  tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['tgt_lang'])

  # Keep 90% for training and 10% for validation
  train_dataset_size = int(0.9 * len(dataset_raw))
  validation_dataset_size = len(dataset_raw) - train_dataset_size
  train_dataset_raw, validation_dataset_raw = random_split(dataset_raw, [train_dataset_size, validation_dataset_size])

  train_dataset = BilingualDataset(train_dataset_raw,
                                   tokenizer_src=tokenizer_src,
                                   tokenizer_tgt=tokenizer_tgt,
                                   src_lang=config['src_lang'],
                                   tgt_lang=config['tgt_lang'],
                                   seq_len=config['seq_len'])

  validation_dataset = BilingualDataset(validation_dataset_raw,
                                        tokenizer_src=tokenizer_src,
                                        tokenizer_tgt=tokenizer_tgt,
                                        src_lang=config['src_lang'],
                                        tgt_lang=config['tgt_lang'],
                                        seq_len=config['seq_len'])

  max_len_src = 0
  max_len_tgt = 0

  for item in dataset_raw:
    src_ids = tokenizer_src.encode(item['translation'][config['src_lang']]).ids
    tgt_ids = tokenizer_src.encode(item['translation'][config['tgt_lang']]).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f'Max Length of Source Sentence: {max_len_src}')
  print(f'Max Length of Target Sentence: {max_len_tgt}')

  train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
  validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

  return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt