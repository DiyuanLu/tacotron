# -*- coding: utf-8 -*-
#/usr/bin/python2

# version to deal with spectrogram
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import modules as mod
#from prepro import load_vocab
import tensorflow as tf
import ipdb

def encode(inputs, is_training=True, scope="encoder", reuse=None):
    ''' 
    Args:
      inputs: A 2d tensor with shape of [N, T], dtype of int32. N: batch_size  T: real length
      seqlens: A 1d tensor with shape of [N,], dtype of int32.
      masks: A 3d tensor with shape of [N, T, 1], dtype of float32.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns: E is the spectrogram filter N
      A collection of Hidden vectors, whose shape is (N, T, E). N seqs, each with T characters, and each of them encoded to E dimension latent representation
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Load vocabulary 
        #char2idx, idx2char = load_vocab()
        
        # Character Embedding  N seqs 
        #inputs = mod.embed(inputs, len(char2idx), hp.embed_size) # (N, T, E) shape=(32, ?, 256)
        # Encoder pre-net: dense(E)--dropout--dense(E/2)--dropout
        #ipdb.set_trace()
        inputs = mod.pre_spectro(inputs, is_training=is_training)   # (N, T, E)
        prenet_out = mod.prenet(inputs, is_training=is_training) # (N, T, E/2)
        
        # Encoder CBHG 
        ## Conv1D bank 
        enc = mod.conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training) # (N, T, K * E / 2)
        
        ### Max pooling
        enc = tf.layers.max_pooling1d(enc, 2, 1, padding="same")  # (N, T, K * E / 2)
          
        ### Conv1D projections
        enc = mod.conv1d(enc, hp.embed_size//2, 3, scope="conv1d_1") # (N, T, E/2)
        enc = mod.normalize(enc, type=hp.norm_type, is_training=is_training, 
                            activation_fn=tf.nn.relu, scope="norm1")
        enc = mod.conv1d(enc, hp.embed_size//2, 3, scope="conv1d_2") # (N, T, E/2)
        enc = mod.normalize(enc, type=hp.norm_type, is_training=is_training, 
                            activation_fn=None, scope="norm2")
        enc += prenet_out # (N, T, E/2) # residual connections
          
        ### Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = mod.highwaynet(enc, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T, E/2)

        ### Bidirectional GRU---apply nonlineararity

        memory = mod.gru(enc, hp.embed_size//2, False) # (N, T, E)  what the network represent the input text input
    
    return memory
        
def decode1(decoder_inputs, memory, is_training=True, scope="decoder1", reuse=None):
    '''
    Args:
      decoder_inputs: A 3d tensor with shape of [N, T', C'], where C'=hp.n_mels*hp.r, 
        dtype of float32. Shifted melspectrogram of sound files. 
      memory: A 3d tensor with shape of [N, T, C], where C=hp.embed_size.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted melspectrogram tensor with shape of [N, T', C'].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        #ipdb.set_trace()
        dec = mod.prenet(decoder_inputs, is_training=is_training) # (N, T', E/2)
        
        # Attention RNN
        dec = mod.attention_decoder(dec, memory, num_units=hp.embed_size) # (N, T', E)

        # Decoder RNNs
        dec += mod.gru(dec, hp.embed_size, False, scope="decoder_gru1") # (N, T', E)
        dec += mod.gru(dec, hp.embed_size, False, scope="decoder_gru2") # (N, T', E)
          
        # Outputs => (N, T', hp.n_mels*hp.r)
        out_dim = decoder_inputs.get_shape().as_list()[-1]
        outputs = tf.layers.dense(dec, out_dim)   # (N, None, E) output the same shape as input
    
    return outputs

def decode2(inputs, is_training=True, scope="decoder2", reuse=None):
    '''
    Args:
      inputs: A 3d tensor with shape of [N, T', C'], where C'=hp.n_mels*hp.r, 
        dtype of float32. Log magnitude spectrogram of sound files.
      is_training: Whether or not the layer is in training mode.  
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted magnitude spectrogram tensor with shape of [N, T', C''], 
        where C'' = (1+hp.n_fft//2)*hp.r.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        prenet_out = mod.prenet(inputs, is_training=is_training) # (N, T'', E/2)
        
        # Decoder Post-processing net = CBHG
        ## Conv1D bank
        dec = mod.conv1d_banks(prenet_out, K=hp.decoder_num_banks, is_training=is_training) # (N, T', E*K/2)
         
        ## Max pooling
        dec = tf.layers.max_pooling1d(dec, 2, 1, padding="same") # (N, T', E*K/2)
         
        ## Conv1D projections
        dec = mod.conv1d(dec, hp.embed_size, 3, scope="conv1d_1") # (N, T', E)
        dec = mod.normalize(dec, type=hp.norm_type, is_training=is_training, 
                            activation_fn=tf.nn.relu, scope="norm1")
        dec = mod.conv1d(dec, hp.embed_size//2, 3, scope="conv1d_2") # (N, T', E/2)
        dec = mod.normalize(dec, type=hp.norm_type, is_training=is_training, 
                            activation_fn=None, scope="norm2")
        dec += prenet_out
         
        ## Highway Nets
        for i in range(4):
            dec = mod.highwaynet(dec, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T, E/2)
         
        ## Bidirectional GRU    
        dec = mod.gru(dec, hp.embed_size//2, True) # (N, T', E)  
        
        # Outputs => (N, T', (1+hp.n_fft//2)*hp.r)
        out_dim = (1+hp.n_fft//2)*hp.r
        outputs = tf.layers.dense(dec, out_dim)
    
    return outputs
