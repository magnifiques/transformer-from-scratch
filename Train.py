import torch
from torch import nn

from pathlib import Path
from Dataset import get_dataset
from Transformer import build_transformer
from torch.utils.tensorboard import SummaryWriter
from Config import get_weights_file_path
from tqdm import tqdm
from Validation import compute_accuracy, run_validation


def get_model(config, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size=src_vocab_size,
                              tgt_vocab_size=tgt_vocab_size,
                              src_seq_len=config['seq_len'],
                              tgt_seq_len=config['seq_len'],
                              d_model=config['d_model'])
    return model

def train_model(config):

  # 1. Define the Device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using Device: {device}')

  # 2. Create the folder
  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

  # 3. Get the DataLoaders, tokenizers, and model
  train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
  model = get_model(config, src_vocab_size=tokenizer_src.get_vocab_size(), tgt_vocab_size=tokenizer_tgt.get_vocab_size()).to(device)

  # # 4. Tensorboard
  writer = SummaryWriter(config['experiment_name'])

  # # 5. Get the Optimizer and Loss Function
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

  initial_step = 0
  global_step = 0

  if config['preload']:
    model_filename = get_weights_file_path(config, config['preload'])
    print(f"Preloading model: {model_filename}")

    state = torch.load(model_filename)
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']

  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

  for epoch in range(initial_step, config['num_epochs']):
    torch.cuda.empty_cache()
    batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch: {epoch:02d}")

    for batch in batch_iterator:
      model.train()

      encoder_input = batch['encoder_input'].to(device) # (batch, seq_len)
      decoder_input = batch['decoder_input'].to(device) # (batch, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (batch, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device) # (batch, 1, seq_len, seq_len)

      # Run the tensors through the transformer
      encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, seq_len, d_model)

      project_output = model.project(decoder_output) # (batch, seq_len, tgt_vocab_size)

      label = batch['label'].to(device) # (batch, seq_len)

      # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, seq_len, tgt_vocab_size)
      loss = loss_fn(project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)).to(device)

      # Compute accuracy
      predictions = project_output.argmax(dim=-1)  # (batch, seq_len)
      accuracy = compute_accuracy(predictions, label, tokenizer_tgt.token_to_id('[PAD]'))

      batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "accuracy": f"{accuracy:.3f}"})

      # Log the Loss
      writer.add_scalar("train_loss", loss.item(), global_step)
      writer.add_scalar("train_accuracy", accuracy, global_step)
      writer.flush()

        

      # Backpropagate the loss
      loss.backward()

      # Update the weights
      optimizer.step()
      optimizer.zero_grad()

      global_step += 1

    run_validation(model,
                   validation_dataloader,
                   tokenizer_src, tokenizer_tgt,
                   config['seq_len'],
                   device,
                   lambda msg: batch_iterator.write(msg),
                   global_step,
                   writer)

    # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config, f"{epoch:02d}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step
    }, model_filename)