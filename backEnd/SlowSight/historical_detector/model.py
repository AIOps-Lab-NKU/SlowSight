import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import math
import time
import warnings
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict

import copy
import json
import os
import stat
from collections import OrderedDict
from itertools import chain

from SlowSight.historical_detector.data_provider.data_factory import data_provider
from SlowSight.historical_detector.layers.Transformer_EncDec import Encoder, EncoderLayer
from SlowSight.historical_detector.layers.SelfAttention_Family import FullAttention, AttentionLayer
from SlowSight.historical_detector.layers.Embed import DataEmbedding_inverted
from SlowSight.historical_detector.utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from SlowSight.historical_detector.utils.lr_schedulers import WarmupPolyLR


def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False


def unfreeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = True


class DropDecoder(nn.Module):
    def __init__(self, d_model, d_ff, seq_len, nhead, num_layers, max_len, p=0.1):
        super().__init__()

        # Create standard Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=p,
            batch_first=True
        )

        # Create full Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, p, max_len)

        # Output projection layer - consistent with original model interface
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d_ff, seq_len)
        )

    def forward(self, z, memory=None):
        # If no memory is provided, use z itself as memory
        # In autoencoder scenarios, z usually comes from encoder output
        if memory is None:
            memory = z

        # Add positional encoding
        z = self.pos_encoder(z)

        # Transformer decoding - no mask to match the original non-directional model behavior
        # You can add appropriate masks if directional generation is required
        decoder_output = self.transformer_decoder(z, memory)

        # Apply output projection to get final result
        w = self.output_projection(decoder_output)

        return w


# Positional encoding implementation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register as buffer instead of parameter
        self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, total_metrics, num_entities, d_model, d_entity, factor, n_heads, d_ff,
                 activation, e_layers, max_len, dropout):
        """
        New parameters explanation:
        - total_metrics: Total number of metrics
        - num_entities: Number of entities
        - d_entity: Dimension of each entity representation in cross-entity attention (can be a hyperparameter)
        """
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_metrics = total_metrics  # Keep track of total metrics count
        self.num_entities = num_entities  # Number of entities
        self.metrics_per_entity = total_metrics // num_entities  # Metrics per entity (should be pre-defined)
        self.d_model = d_model
        self.d_entity = d_entity
        self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.max_len = max_len
        self.dropout = dropout

        # External input embedding module (input shape: (B, T, total_metrics))
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.dropout)

        # Encoder: model the entire concatenated time series, output shape (B, T, d_model)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=True),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )

        # ---------------- New cross-entity module ----------------
        # Map encoder output to (B, T, num_entities * d_entity) space
        self.entity_mapping = nn.Linear(self.d_model, self.num_entities * self.d_entity)
        # Cross-entity attention module: perform interaction among num_entities representations at each time step
        self.cross_entity_attn = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=True),
                        self.d_entity, self.n_heads),
                    self.d_entity,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                )
            ],
            norm_layer=nn.LayerNorm(self.d_entity),
        )
        # Project back to (B, T, d_model)
        self.entity_proj = nn.Linear(self.num_entities * self.d_entity, self.d_model)
        # --------------------------------------------------

        self.decoder_G = DropDecoder(self.d_model, self.d_ff, self.seq_len, self.n_heads, self.e_layers, self.max_len, p=self.dropout)
        self.decoder_D = DropDecoder(self.d_model, self.d_ff, self.seq_len, self.n_heads, self.e_layers, self.max_len, p=self.dropout)

    def apply_cross_entity(self, x):
        """
        Apply cross-entity attention to encoder output x (shape: (B, T, d_model))
        Steps:
          1. Project x using entity_mapping to (B, T, num_entities*d_entity)
          2. Reshape to (B, T, num_entities, d_entity)
          3. Merge B and T → (B*T, num_entities, d_entity), apply cross-entity attention,
             output remains (B*T, num_entities, d_entity)
          4. Restore to (B, T, num_entities, d_entity), then flatten to (B, T, num_entities*d_entity)
          5. Fuse back to (B, T, d_model) via entity_proj
        """
        B, T, _ = x.shape
        # Map and reshape
        x_mapped = self.entity_mapping(x)  # (B, T, num_entities*d_entity)
        x_entities = x_mapped.view(B, T, self.num_entities, self.d_entity)  # (B, T, num_entities, d_entity)
        # Merge B and T for cross-entity attention
        x_entities_flat = x_entities.view(B * T, self.num_entities, self.d_entity)  # (B*T, num_entities, d_entity)
        # Cross-entity attention
        x_attn = self.cross_entity_attn(x_entities_flat)[0]  # (B*T, num_entities, d_entity)
        # Restore to (B, T, num_entities, d_entity)
        x_attn = x_attn.view(B, T, self.num_entities, self.d_entity)
        # Flatten entities: (B, T, num_entities*d_entity)
        x_attn_flat = x_attn.view(B, T, self.num_entities * self.d_entity)
        # Fuse into d_model space
        x_fused = self.entity_proj(x_attn_flat)  # (B, T, d_model)
        return x_fused

    def forward(self, x):
        """
        x: (B, T, total_metrics)
        1. Pass through embedding and encoder to get (B, T, d_model)
        2. Apply cross-entity attention to obtain updated representations (B, T, d_model)
        """
        # Original embedding
        emb = self.enc_embedding(x, None)  # Depends on DataEmbedding_inverted implementation
        enc_out = self.encoder(emb)[0]  # (B, T, d_model)
        # Apply cross-entity attention
        out = self.apply_cross_entity(enc_out)  # (B, T, d_model)
        return out


class TransGAN(object):
    def __init__(self, seq_len, pred_len, total_metrics, num_entities, d_model, d_entity, factor, n_heads, d_ff,
                 activation, e_layers, dropout, train_epochs, learning_rate, patience, weight_decay, max_len, batch_size):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_metrics = total_metrics
        self.num_entities = num_entities
        self.d_model = d_model
        self.d_entity = d_entity
        self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.dropout = dropout
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.batch_size = batch_size

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = Model(self.seq_len, self.pred_len,
                      self.total_metrics,
                      self.num_entities,
                      self.d_model,
                      self.d_entity,
                      self.factor,
                      self.n_heads,
                      self.d_ff,
                      self.activation,
                      self.e_layers,
                      self.max_len,
                      self.dropout).float()
        return model

    def _acquire_device(self):
        device = torch.device('cpu')
        print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(flag, self.train_data, self.valid_data, self.test_data, self.batch_size,
                                              self.seq_len)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def freeze_layers(self):
        for attn_layer in self.model.encoder.attn_layers:
            attn_layer.requires_grad = False
        self.model.decoder_G.pos_encoder.requires_grad = False
        self.model.decoder_G.transformer_decoder.requires_grad = False
        self.model.decoder_D.pos_encoder.requires_grad = False
        self.model.decoder_D.transformer_decoder.requires_grad = False

    def unfreeze_layers(self):
        for attn_layer in self.model.encoder.attn_layers:
            attn_layer.requires_grad = True
        self.model.decoder_G.pos_encoder.requires_grad = True
        self.model.decoder_G.transformer_decoder.requires_grad = True
        self.model.decoder_D.pos_encoder.requires_grad = True
        self.model.decoder_D.transformer_decoder.requires_grad = True

    def init_layers(self):
        self.model.decoder_G.output_projection[0].reset_parameters()
        self.model.decoder_G.output_projection[3].reset_parameters()
        self.model.decoder_D.output_projection[0].reset_parameters()
        self.model.decoder_D.output_projection[3].reset_parameters()

    def processed_data(self, train_data, valid_data, test_data):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

    def attention_entropy(self, attn_matrix):
        """Calculate Transformer attention entropy"""
        entropy = - (attn_matrix * torch.log(attn_matrix + 1e-8)).sum(dim=-1)  # Avoid log(0)
        entropy = entropy.mean(dim=1)  # Mean across attention heads -> [B, L]
        return entropy  # [B, L]

    def compute_anomaly_score(self, entropy_score, dim=2):
        """Compute group anomaly score for each disk"""
        H_avg = torch.mean(entropy_score, dim=dim, keepdim=True)  # Compute entropy mean across disks
        anomaly_score = torch.abs(entropy_score - H_avg)  # Compute deviation
        return anomaly_score  # [B, L, N]

    @staticmethod
    def get_model_gradients(model):
        grads = OrderedDict()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad
        return grads

    @staticmethod
    def update_grads(model, grads):
        for name, param in model.named_parameters():
            if name in grads:
                param.grad = param.grad + grads[name]

    def g_parameters(self):
        return chain(self.model.encoder.parameters(), self.model.decoder_G.parameters())

    def d_parameters(self):
        return chain(self.model.encoder.parameters(), self.model.decoder_D.parameters())

    def parameters_all(self):
        return chain(self.model.encoder.parameters(), self.model.decoder_G.parameters(),
                      self.model.decoder_D.parameters())

    def fit(self):
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='valid')

        path = './checkpoints/'
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping1 = EarlyStopping(patience=self.patience, verbose=True)
        early_stopping2 = EarlyStopping(patience=self.patience, verbose=True)
        model_optim = torch.optim.AdamW(self.parameters_all(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = self._select_criterion()
        scheduler_all = WarmupPolyLR(model_optim, self.train_epochs, warmup_iters=5)

        train_loss1 = []
        train_loss2 = []
        valid_loss1 = []
        valid_loss2 = []
        train_time = 0
        valid_time = 0

        for epoch in range(self.train_epochs):
            train_start = time.time()
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            train_losses1 = []
            train_losses2 = []

            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)  # batch_x: (B, T, total_metrics)
                # ------------------ Modified flow: keep input structure unchanged ------------------
                # Through embedding, encoder, and cross-entity attention
                z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
                z = self.model.apply_cross_entity(z)
                # --------------------------------------------------------------------------

                w_G = self.model.decoder_G(z).permute(0, 2, 1)
                w_D = self.model.decoder_D(z).permute(0, 2, 1)
                w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2, 1)

                loss_G = (1 / (epoch + 1)) * criterion(w_G, batch_x) + (1 - 1 / (epoch + 1)) * criterion(w_G_D, batch_x)
                loss_D = (1 / (epoch + 1)) * criterion(w_D, batch_x) - (1 - 1 / (epoch + 1)) * criterion(w_G_D, batch_x)

                model_optim.zero_grad()
                loss_G.backward(retain_graph=True)
                g_encoder_grads = self.get_model_gradients(self.model.encoder)
                loss_D.backward()
                self.update_grads(self.model.encoder, g_encoder_grads)
                model_optim.step()

                train_losses1.append(loss_G.detach().numpy())
                train_losses2.append(loss_D.detach().numpy())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss1: {2:.7f} | loss2: {3:.7f}".format(i + 1, epoch + 1,
                                                                                              loss_G.item(),
                                                                                              loss_D.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            scheduler_all.step()
            train_loss1_mean = np.mean(train_losses1)
            train_loss2_mean = np.mean(train_losses2)
            train_loss1.append(train_loss1_mean)
            train_loss2.append(train_loss2_mean)
            train_time += time.time() - train_start
            print("Epoch: {0} | Train Loss1: {1:.7f} Train Loss2: {2:.7f}".format(epoch + 1, train_loss1_mean,
                                                                                  train_loss2_mean))
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            val_losses1 = []
            val_losses2 = []
            valid_start = time.time()
            for i, batch_x in enumerate(valid_loader):
                batch_x = batch_x.float().to(self.device)
                # Same flow as training: through embedding, encoder, and cross-entity attention
                z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
                z = self.model.apply_cross_entity(z)
                w_G = self.model.decoder_G(z).permute(0, 2, 1)
                w_D = self.model.decoder_D(z).permute(0, 2, 1)
                w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2, 1)

                val_loss1 = 1 / (epoch + 1) * criterion(batch_x, w_G) + (1 - 1 / (epoch + 1)) * criterion(batch_x, w_G_D)
                val_loss2 = 1 / (epoch + 1) * criterion(batch_x, w_D) - (1 - 1 / (epoch + 1)) * criterion(batch_x, w_G_D)
                val_losses1.append(val_loss1.detach().numpy())
                val_losses2.append(val_loss2.detach().numpy())

            valid_time += time.time() - valid_start
            val1_loss = np.mean(val_losses1)
            val2_loss = np.mean(val_losses2)
            valid_loss1.append(val1_loss)
            valid_loss2.append(val2_loss)
            print("Epoch: {0} | Vali Loss1: {1:.7f} Vali Loss2: {2:.7f}".format(epoch + 1, val1_loss.item(),
                                                                                val2_loss.item()))
            early_stopping1(np.average(valid_loss1), self.model, path)
            early_stopping2(np.average(valid_loss2), self.model, path)
            if early_stopping1.early_stop and early_stopping2.early_stop:
                print("Early stopping")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return {
            'epoch': epoch + 1,
            'train_time': train_time,
            'valid_time': valid_time,
            'train_loss1': np.array(train_loss1).tolist(),
            'train_loss2': np.array(train_loss2).tolist(),
            'valid_loss1': np.array(valid_loss1).tolist(),
            'valid_loss2': np.array(valid_loss2).tolist()
        }

    def predict(self, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        valid_data, valid_loader = self._get_data(flag='valid')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/', 'checkpoint.pth')))

        attens_energy = []

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        train_reconstruct_error = []
        train_model_processed_data = []
        train_model_reconstruct_data = []

        valid_reconstruct_error = []
        valid_model_processed_data = []
        valid_model_reconstruct_data = []

        # (1) Statistics on train set
        with torch.no_grad():
            for i, batch_x in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
                z = self.model.apply_cross_entity(z)
                w_G = self.model.decoder_G(z).permute(0, 2, 1)
                w_D = self.model.decoder_D(z).permute(0, 2, 1)
                w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2,
                                                                                                                 1)
                reconstruct_error = 0.1 * np.square(
                    batch_x[:, -1, :].detach().numpy() - w_G[:, -1, :].detach().numpy()) + \
                                   (1 - 0.1) * np.square(
                    batch_x[:, -1, :].detach().numpy() - w_G_D[:, -1, :].detach().numpy())

                score = np.sum(reconstruct_error, axis=-1)
                attens_energy.extend(score)

                train_model_processed_data.append(batch_x[:, -1, :].detach().numpy())
                train_model_reconstruct_data.append(w_G_D[:, -1, :].detach().numpy())

                train_reconstruct_error.extend(reconstruct_error)

            train_model_processed_data = np.concatenate(train_model_processed_data, axis=0)
            train_model_reconstruct_data = np.concatenate(train_model_reconstruct_data, axis=0)

        train_energy = np.array(attens_energy)

        attens_energy = []
        with torch.no_grad():
            for i, batch_x in enumerate(valid_loader):
                batch_x = batch_x.float().to(self.device)
                z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
                z = self.model.apply_cross_entity(z)
                w_G = self.model.decoder_G(z).permute(0, 2, 1)
                w_D = self.model.decoder_D(z).permute(0, 2, 1)
                w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2,
                                                                                                                 1)
                reconstruct_error = 0.1 * np.square(
                    batch_x[:, -1, :].detach().numpy() - w_G[:, -1, :].detach().numpy()) + \
                                   (1 - 0.1) * np.square(
                    batch_x[:, -1, :].detach().numpy() - w_G_D[:, -1, :].detach().numpy())

                score = np.sum(reconstruct_error, axis=-1)
                attens_energy.extend(score)

                valid_model_processed_data.append(batch_x[:, -1, :].detach().numpy())
                valid_model_reconstruct_data.append(w_G_D[:, -1, :].detach().numpy())

                valid_reconstruct_error.extend(reconstruct_error)

            valid_model_processed_data = np.concatenate(valid_model_processed_data, axis=0)
            valid_model_reconstruct_data = np.concatenate(valid_model_reconstruct_data, axis=0)

        valid_energy = np.array(attens_energy)

        test_reconstruct_error = []
        test_model_processed_data = []
        test_model_reconstruct_data = []

        attens_energy = []
        test_labels = []

        for i, batch_x in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            z = self.model.encoder(self.model.enc_embedding(batch_x, None))[0]
            z = self.model.apply_cross_entity(z)
            w_G = self.model.decoder_G(z).permute(0, 2, 1)
            w_D = self.model.decoder_D(z).permute(0, 2, 1)
            w_G_D = self.model.decoder_D(self.model.encoder(self.model.enc_embedding(w_G, None))[0]).permute(0, 2, 1)

            # criterion
            reconstruct_error = 0.1 * np.square(batch_x[:, -1, :].detach().numpy() - w_G[:, -1, :].detach().numpy()) + \
                                (1 - 0.1) * np.square(
                batch_x[:, -1, :].detach().numpy() - w_G_D[:, -1, :].detach().numpy())

            score = np.sum(reconstruct_error, axis=-1)
            attens_energy.extend(score)
            # test_labels.extend(batch_y[:, -1, -1].detach().numpy())
            # test_labels.extend(batch_y[:, -1].detach().numpy())

            test_model_processed_data.append(batch_x[:, -1, :].detach().numpy())
            test_model_reconstruct_data.append(w_G_D[:, -1, :].detach().numpy())

            test_reconstruct_error.extend(reconstruct_error)

        test_model_processed_data = np.concatenate(test_model_processed_data, axis=0)
        test_model_reconstruct_data = np.concatenate(test_model_reconstruct_data, axis=0)

        test_energy = np.array(attens_energy)

        return {
            'train_predict_G': train_model_processed_data,
            'train_predict_G_D': train_model_reconstruct_data,
            'valid_predict_G': valid_model_processed_data,
            'valid_predict_G_D': valid_model_reconstruct_data,
            'test_predict_G': test_model_processed_data,
            'test_predict_G_D': test_model_reconstruct_data
        }

    def save(self, encoder_path, decoder_G_path, decoder_D_path):
        """
        Save weights to specified file paths
        """
        torch.save(self.model.encoder.state_dict(), encoder_path)
        torch.save(self.model.decoder_G.state_dict(), decoder_G_path)
        torch.save(self.model.decoder_D.state_dict(), decoder_D_path)

    def restore(self, encoder_path, decoder_G_path, decoder_D_path):
        """
        Load weights from file paths
        """
        self.model.encoder.load_state_dict(torch.load(encoder_path))
        self.model.decoder_G.load_state_dict(torch.load(decoder_G_path))
        self.model.decoder_D.load_state_dict(torch.load(decoder_D_path))

    def save_weights(self):
        """
        Save weights and return as tuple
        """
        encoder = self.model.encoder.state_dict()
        decoder_D = self.model.decoder_D.state_dict()
        decoder_G = self.model.decoder_G.state_dict()
        return (encoder, decoder_D, decoder_G)

    def restore_weights(self, encoder, decoder_D, decoder_G):
        """
        Load weights from passed parameters
        """
        self.model.encoder.load_state_dict(encoder)
        self.model.decoder_D.load_state_dict(decoder_D)
        self.model.decoder_G.load_state_dict(decoder_G)
