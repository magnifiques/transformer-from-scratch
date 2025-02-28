import torch
from Dataset import causal_mask


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):

  sos_index = tokenizer_src.token_to_id('[SOS]')
  eos_index = tokenizer_src.token_to_id('[EOS]')

  # Precompute the encoder output and reuse it for every token we get from the decoder
  encoder_output = model.encode(source, source_mask)

  # Initialize the decoder input with the SOS token
  decoder_input = torch.empty(1, 1).fill_(sos_index).type_as(source).to(device)

  while True:
    if decoder_input.size(1) == max_len:
      break

    # Build mask for the Target (Decoder Input)
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

    # Calculate the output of the decoder
    out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

    # Get the Next Token
    prob = model.project(out[:, -1])

    # Select the token with the max probability (Because it is Greedy Search)
    _, next_word = torch.max(prob, dim=1)
    decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

    if next_word == eos_index:
      break


  return decoder_input.squeeze(0)

def run_validation(model,
                   validation_dataset,
                   tokenizer_src,
                   tokenizer_tgt,
                   max_len,
                   device,
                   print_msg,
                   global_state,
                   writer,
                   num_examples = 2):

  model.eval()
  count = 0

  source_texts = []
  expected = []
  predicted = []

  # Size of the control window (Just Use the default value)

  control_width = 80

  with torch.no_grad():
    for batch in validation_dataset:
      count += 1
      encoder_input = batch['encoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)

      assert encoder_input.size(0) == 1, "Batch Size must be 1 for validation"

      model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

      source_text = batch['src_text'][0]
      target_text = batch['tgt_text'][0]
      model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

      # Print the results
      print_msg('-'*control_width)
      print_msg(f"Source Text: {source_text}")
      print_msg(f"Target Text: {target_text}")
      print_msg(f"Predicted Text: {model_out_text}")

      if count == num_examples:
        break
      
def compute_accuracy(predictions, labels, pad_token_id):
    """
    Computes the accuracy of predictions.
    Ignores padding tokens in the calculation.
    """
    non_pad_mask = labels.ne(pad_token_id)  # Mask out [PAD] tokens
    correct = predictions.eq(labels) & non_pad_mask  # Only count correct predictions that are not padding
    acc = correct.sum().item() / non_pad_mask.sum().item()  # Normalize by valid tokens
    return acc
