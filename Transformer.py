import torch
from torch import nn
from Embedding import InputEmbedding, PositionalEmbedding
import MultiHeadAttentionBlock
from FeedForward import FeedForwardBlock
from Encoder import Encoder, EncoderBlock
from Decoder import Decoder, DecoderBlock
import ProjectionLayer

class Transformer(nn.Module):

  def __init__(self,
               src_embed: InputEmbedding,
               target_embed: InputEmbedding,
               src_pos: PositionalEmbedding,
               target_pos: PositionalEmbedding,
               encoder: Encoder,
               decoder: Decoder,
               projection_layer: ProjectionLayer) -> None:
    super().__init__()
    self.src_embed = src_embed
    self.target_embed = target_embed
    self.src_pos = src_pos
    self.target_pos = target_pos
    self.encoder = encoder
    self.decoder = decoder
    self.projection_layer = projection_layer

  def encode(self, src, src_mask):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)

  def decode(self, encoder_output, src_mask, tgt, tgt_mask):
    tgt = self.target_embed(tgt)
    tgt = self.target_pos(tgt)
    return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

  def project(self, x):
    return self.projection_layer(x)

def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      heads: int = 8,
                      number_of_blocks: int = 6,
                      dropout: float = 0.1,
                      d_ff: int = 2048) -> Transformer:

  # Create the Input Embedding Layers
  src_embed = InputEmbedding(d_model, src_vocab_size)
  tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

  # Create the Positional Embedding Layers
  src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
  tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

  # Create the Encoder Blocks
  encoder_blocks = []
  for _ in range(number_of_blocks):
    encoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
    encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)

  # Create the Decoder Blocks
  decoder_blocks = []
  for _ in range(number_of_blocks):
    decoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
    decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
    decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
    decoder_blocks.append(decoder_block)

  # Create the Encoder and Decoder
  encoder = Encoder(nn.ModuleList(encoder_blocks))
  decoder = Decoder(nn.ModuleList(decoder_blocks))

  # Create the Projection Layers
  projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

  # Create the Transformer
  transformer = Transformer(src_embed=src_embed,
                            target_embed=tgt_embed,
                            src_pos=src_pos,
                            target_pos=tgt_pos,
                            encoder=encoder,
                            decoder=decoder,
                            projection_layer=projection_layer)

  # Initialize the Parameters
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer
