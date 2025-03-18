#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare Multiple Methods (MHENST-DQN, VanillaDQN, etc.) across multiple environments and seeds,
logging and plotting per-episode metrics:
 - Reward
 - Epsilon
 - Memory usage
 - VAE reconstruction error
 - Average Q-value

All environments run for 2000 episodes by default.
If minigrid is installed, it also compares on MiniGrid-DoorKey.
Early stopping is optional (but off by default here),
multi-seed statistical comparison, and PDF output with mean ± std curves are included.
"""

import os
import gc
import copy
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Any, Dict
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import ttest_ind

# Torch with new AMP API
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

import gymnasium as gym

# (Optional) MiniGrid environment
try:
    from minigrid.wrappers import ImgObsWrapper
    from minigrid.envs import DoorKeyEnv
    MINIGRID_AVAILABLE = True
except ImportError:
    MINIGRID_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == 'cuda')
print(f"[INFO] Using device={DEVICE}, USE_AMP={USE_AMP}")

###############################################################################
# 1. Environment Adapter
###############################################################################
class EnvAdapter:
    """Flattens environment observations into float32 arrays."""
    def __init__(self, env):
        self.env = env

    def preprocess_observation(self, obs):
        # Example for multi-modal puzzle (dict with "vars", "image", "text_token")
        if isinstance(obs, dict):
            out_vars = obs["vars"].astype(np.float32)
            image    = obs["image"].astype(np.float32)
            text_tok = obs["text_token"]
            flat_img = image.flatten()
            text_oh  = np.zeros(4, dtype=np.float32)
            text_oh[text_tok] = 1.0
            return np.concatenate([out_vars, flat_img, text_oh], axis=0)
        if hasattr(obs, 'shape') and len(obs.shape)>1:
            return obs.astype(np.float32).flatten()
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)
        return np.array([obs], dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        return self.preprocess_observation(obs), info

    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        done = bool(done or trunc)
        obs_p= self.preprocess_observation(obs)
        return obs_p, rew, done, info

    def close(self):
        self.env.close()

###############################################################################
# 2. EarlyStopping Monitor (if needed)
###############################################################################
class EarlyStoppingMonitor:
    """
    Checks a chosen metric (e.g. reward) for improvement over patience steps;
    triggers early stop if no improvement.
    mode='max' => we want metric to go up
    mode='min' => we want metric to go down
    """
    def __init__(self, patience=200, mode='max'):
        self.patience = patience
        self.mode     = mode
        self.best_value = None
        self.wait_count = 0

    def check(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False
        improved= False
        if self.mode=='max' and current_value> self.best_value:
            improved= True
        elif self.mode=='min' and current_value< self.best_value:
            improved= True

        if improved:
            self.best_value= current_value
            self.wait_count= 0
            return False
        else:
            self.wait_count+=1
            if self.wait_count>= self.patience:
                return True
            return False

###############################################################################
# 3. Core MHENST-DQN Implementation
###############################################################################
from collections import namedtuple
Transition= namedtuple('Transition', ('state','action','reward','next_state','done'))

class SequenceReplayBuffer:
    def __init__(self, capacity=2000, seq_len=4, alpha=0.6):
        self.capacity= capacity
        self.seq_len= seq_len
        self.buffer= deque(maxlen=capacity)
        self.episode_buffer= []
        self.alpha= alpha
        self.priorities= np.ones((capacity,), dtype=np.float32)
        self.position= 0
        self.is_full= False

    def add_transition(self, state, action, reward, next_state, done):
        self.episode_buffer.append(Transition(state, action, reward, next_state, done))
        if done:
            self._finish_episode()

    def _finish_episode(self):
        ep_len= len(self.episode_buffer)
        if ep_len==0: return
        if ep_len< self.seq_len:
            pad= [self.episode_buffer[0]]*(self.seq_len- ep_len)+ self.episode_buffer
            self._add_sequence(pad[-self.seq_len:])
        else:
            stride= max(1, min(3, ep_len//4))
            for i in range(0, ep_len- self.seq_len+1, stride):
                seq= self.episode_buffer[i:i+self.seq_len]
                self._add_sequence(seq)
        self.episode_buffer= []

    def _add_sequence(self, seq):
        max_p= np.max(self.priorities[:len(self.buffer)]) if len(self.buffer)>0 else 1.0
        if self.is_full:
            idx_min= np.argmin(self.priorities)
            self.buffer[idx_min]= seq
            self.priorities[idx_min]= max_p
        else:
            if len(self.buffer)< self.capacity:
                self.buffer.append(seq)
            else:
                self.buffer[self.position]= seq
            self.priorities[self.position]= max_p
            self.position= (self.position+1)% self.capacity
            if self.position==0:
                self.is_full= True

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer)==0:
            return None,None,None
        valid_sz= len(self.buffer)
        batch_size= min(batch_size, valid_sz)
        p= self.priorities[:valid_sz]** self.alpha
        p/= p.sum()
        idxs= np.random.choice(valid_sz, batch_size, replace=False, p=p)
        weights= (valid_sz*p[idxs])**(-beta)
        weights/= weights.max()
        samples= [self.buffer[i] for i in idxs]
        return samples, idxs, weights

    def update_priorities(self, idxs, td_errs):
        for idx, err in zip(idxs, td_errs):
            if idx< len(self.priorities):
                self.priorities[idx]= abs(err)+ 1e-6

    def __len__(self):
        return len(self.buffer)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EncoderBlock(nn.Module):
    def __init__(self, in_dim,hid_dim):
        super().__init__()
        self.fc1= nn.Linear(in_dim,hid_dim)
        self.fc2= nn.Linear(hid_dim,hid_dim)
        self.ln= nn.LayerNorm(hid_dim)
    def forward(self,x):
        x= F.relu(self.fc1(x))
        x= self.fc2(x)
        return self.ln(x)

class VariationalEncoder(nn.Module):
    def __init__(self,in_dim,hid_dim,lat_dim):
        super().__init__()
        self.enc_block= EncoderBlock(in_dim,hid_dim)
        self.fc_mu= nn.Linear(hid_dim, lat_dim)
        self.fc_logv= nn.Linear(hid_dim, lat_dim)
    def forward(self,x):
        h= self.enc_block(x)
        mu= self.fc_mu(h)
        logv= self.fc_logv(h)
        if self.training:
            std= torch.exp(0.5* logv)
            eps= torch.randn_like(std)
            z= mu+ eps*std
        else:
            z= mu
        return z, mu, logv

class Decoder(nn.Module):
    def __init__(self, lat_dim,hid_dim,out_dim):
        super().__init__()
        self.fc1= nn.Linear(lat_dim,hid_dim)
        self.fc2= nn.Linear(hid_dim,hid_dim)
        self.fc3= nn.Linear(hid_dim,out_dim)
        self.ln= nn.LayerNorm(hid_dim)
    def forward(self,z):
        h= F.relu(self.fc1(z))
        h= F.relu(self.fc2(h))
        h= self.ln(h)
        return self.fc3(h)

class VAEModule(nn.Module):
    def __init__(self,in_dim,hid_dim,lat_dim):
        super().__init__()
        self.encoder= VariationalEncoder(in_dim,hid_dim,lat_dim)
        self.decoder= Decoder(lat_dim,hid_dim,in_dim)

    def forward(self,x):
        z,mu,logv= self.encoder(x)
        xr= self.decoder(z)
        return xr,z,mu,logv

    def encode(self,x):
        z, _, _= self.encoder(x)
        return z

    def compute_loss(self,x,xr,mu,logv):
        recon= F.mse_loss(xr,x, reduction='sum')
        kl   = -0.5* torch.sum(1+ logv- mu.pow(2)- logv.exp())
        return recon+ 0.1* kl

# Torch Geometric
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class KnowledgeGraphGNN(nn.Module):
    """Graph-based module for neural-symbolic integration."""
    def __init__(self,node_feature_dim, hidden_dim=16, output_dim=8, node_count=8):
        super().__init__()
        self.num_nodes= node_count
        self.node_features= nn.Parameter(torch.randn(node_count,node_feature_dim)*0.1)
        self.conv1= GCNConv(node_feature_dim, hidden_dim)
        self.conv2= GCNConv(hidden_dim, output_dim)
        edges=[]
        for i in range(node_count):
            for j in range(node_count):
                if i!=j:
                    edges.append([i,j])
        self.edge_index= torch.tensor(edges,dtype=torch.long).t().contiguous()
        self.edge_attr= nn.Parameter(torch.ones(self.edge_index.size(1),1))

    def update_graph_with_latent(self, latent_vec):
        bsz= latent_vec.size(0)
        base= self.node_features.unsqueeze(0).expand(bsz,-1,-1)
        w= torch.softmax(latent_vec[:, :self.num_nodes], dim=1).unsqueeze(-1)
        return base + 0.1*w

    def forward(self, latent_vec):
        self.edge_index= self.edge_index.to(latent_vec.device)
        self.edge_attr= self.edge_attr.to(latent_vec.device)
        bsz= latent_vec.size(0)
        outs=[]
        node_batch= self.update_graph_with_latent(latent_vec)
        for i in range(bsz):
            x= node_batch[i]
            data= Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)
            data= data.to(latent_vec.device)
            x1= F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
            x1= F.dropout(x1,p=0.1,training=self.training)
            x2= self.conv2(x1, data.edge_index, data.edge_attr)
            emb= torch.mean(x2, dim=0)
            outs.append(emb)
        return torch.stack(outs, dim=0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq=100):
        super().__init__()
        pe= torch.zeros(max_seq, d_model)
        position= torch.arange(0,max_seq,dtype=torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
        pe[:,0::2]= torch.sin(position*div_term)
        pe[:,1::2]= torch.cos(position*div_term)
        pe= pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        sl= x.size(1)
        return x+ self.pe[:, :sl, :]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.attn= nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ff= nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.norm1= nn.LayerNorm(embed_dim)
        self.norm2= nn.LayerNorm(embed_dim)
        self.drop= nn.Dropout(dropout)

    def forward(self,x,mask=None):
        a,_= self.attn(x,x,x, attn_mask=mask)
        x= x+ self.drop(a)
        x= self.norm1(x)
        f= self.ff(x)
        x= x+ self.drop(f)
        x= self.norm2(x)
        return x

class EnhancedTransformerDQN(nn.Module):
    def __init__(self, input_dim, action_dim, seq_len=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.action_dim= action_dim
        self.embed_dim= input_dim
        self.embedding= nn.Sequential(
            nn.Linear(input_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )
        self.posenc= PositionalEncoding(self.embed_dim, max_seq=seq_len)
        self.blocks= nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads=1, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.adv_stream= nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim//2),
            nn.ReLU(),
            nn.Linear(self.embed_dim//2, action_dim)
        )
        self.val_stream= nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim//2),
            nn.ReLU(),
            nn.Linear(self.embed_dim//2, 1)
        )

    def forward(self,x):
        bsz, sl, _= x.shape
        x= self.embedding(x)
        x= self.posenc(x)
        x= x.transpose(0,1)
        for blk in self.blocks:
            x= blk(x)
        x= x.transpose(0,1)
        out_list=[]
        for t in range(sl):
            feat_t= x[:, t, :]
            adv= self.adv_stream(feat_t)
            val= self.val_stream(feat_t)
            q_t= val+ adv - adv.mean(dim=1, keepdim=True)
            out_list.append(q_t.unsqueeze(1))
        return torch.cat(out_list, dim=1)

class EnhancedMetaCognition:
    def __init__(self, init_epsilon=1.0, min_epsilon=0.05, eps_decay=0.995,
                 init_lr=1e-3, min_lr=1e-5, lr_decay=0.9999,
                 reward_threshold=0.5):
        self.epsilon= init_epsilon
        self.min_epsilon= min_epsilon
        self.eps_decay= eps_decay
        self.lr= init_lr
        self.min_lr= min_lr
        self.lr_decay= lr_decay
        self.reward_threshold= reward_threshold

        self.reward_history=[]
        self.epsilon_history=[]
        self.lr_history=[]
        self.step_count=0

    def update_epsilon(self, rew=None):
        if rew is not None:
            self.reward_history.append(rew)
        if len(self.reward_history)>=10:
            avg_r10= np.mean(self.reward_history[-10:])
            if avg_r10< self.reward_threshold:
                self.epsilon= min(self.epsilon*1.01, 0.8)
            else:
                self.epsilon= max(self.min_epsilon, self.epsilon*self.eps_decay)
        else:
            self.epsilon= max(self.min_epsilon, self.epsilon*self.eps_decay)
        self.epsilon_history.append(self.epsilon)
        return self.epsilon

    def update_learning_rate(self, optimizer, loss_val=None):
        self.step_count+=1
        if self.step_count%50==0:
            new_lr= max(self.min_lr, self.lr*self.lr_decay)
            if loss_val is not None and loss_val>1e3:
                new_lr= max(self.min_lr, new_lr*0.9)
            for pg in optimizer.param_groups:
                pg['lr']= new_lr
            self.lr= new_lr
            self.lr_history.append(new_lr)

    def select_action(self, seq_input, dqn):
        if random.random()< self.epsilon:
            return random.randint(0, dqn.action_dim-1)
        else:
            with torch.no_grad():
                q_all= dqn(seq_input)
                q_last= q_all[:,-1,:]
                act= int(torch.argmax(q_last,dim=1).item())
            return act


###############################################################################
# 4) Training Helper Functions
###############################################################################
def train_vae_step_collect(vae, states_array, optimizer, scaler, ep_vae_losses):
    """Train the VAE and log reconstruction error to ep_vae_losses."""
    vae.train()
    if not isinstance(states_array, torch.Tensor):
        states_array= torch.tensor(states_array,dtype=torch.float32,device=DEVICE)
    optimizer.zero_grad()
    with autocast(device_type='cuda', enabled=(DEVICE.type=='cuda')):
        xr,z,mu,logv= vae(states_array)
        loss= vae.compute_loss(states_array, xr, mu, logv)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        recon_mse= F.mse_loss(xr, states_array, reduction='mean').item()
        ep_vae_losses.append(recon_mse)
    return loss.item()


def train_dqn_step_collect(dqn_online, dqn_target,
                           vae, gnn,
                           sequences,
                           dqn_opt, scaler_dqn,
                           meta, idxs, replay_buf,
                           ep_q_values):
    """One step of DQN training from a batch of sequences."""
    td_errs=[]
    for seq in sequences:
        s_buf, a_buf, r_buf, ns_buf, d_buf= [],[],[],[],[]
        for tr in seq:
            s_buf.append(tr.state)
            a_buf.append(tr.action)
            r_buf.append(tr.reward)
            ns_buf.append(tr.next_state)
            d_buf.append(tr.done)

        s_buf= torch.tensor(np.array(s_buf,dtype=np.float32),device=DEVICE)
        a_buf= torch.tensor(a_buf, dtype=torch.long, device=DEVICE)
        r_buf= torch.tensor(r_buf, dtype=torch.float32, device=DEVICE)
        ns_buf= torch.tensor(np.array(ns_buf,dtype=np.float32),device=DEVICE)
        d_buf= torch.tensor(d_buf, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            z_s= vae.encode(s_buf)
            z_ns= vae.encode(ns_buf)
            emb_s= gnn(z_s)
            emb_ns= gnn(z_ns)
        emb_s= emb_s.unsqueeze(0)
        emb_ns= emb_ns.unsqueeze(0)

        with autocast(device_type='cuda', enabled=(DEVICE.type=='cuda')):
            q_all= dqn_online(emb_s)
            q_last= q_all[:,-1,:]
            act_last= a_buf[-1]
            rew_last= r_buf[-1]
            done_last= d_buf[-1]

            q_pred= q_last[0, act_last]
            q_next= dqn_target(emb_ns)
            q_last_ns= q_next[:,-1,:]
            q_next_max= q_last_ns.max(dim=1)[0]
            gamma= 0.99
            q_targ= rew_last + (1.- done_last)* gamma* q_next_max[0]

            td_err= (q_pred- q_targ).detach().cpu().item()
            td_errs.append(td_err)
            loss_dqn= F.smooth_l1_loss(q_pred.unsqueeze(0), q_targ.unsqueeze(0))

        dqn_opt.zero_grad()
        scaler_dqn.scale(loss_dqn).backward()
        scaler_dqn.step(dqn_opt)
        scaler_dqn.update()

        meta.update_learning_rate(dqn_opt, loss_dqn.item())

        with torch.no_grad():
            avgQ= q_last.mean().item()
            ep_q_values.append(avgQ)

    replay_buf.update_priorities(idxs, td_errs)
    # soft update
    for p_t, p_o in zip(dqn_target.parameters(), dqn_online.parameters()):
        p_t.data.copy_(0.01* p_o.data + 0.99* p_t.data)


###############################################################################
# 5) run_mhenst_dqn_on_env (Finally!)
###############################################################################
def run_mhenst_dqn_on_env(
        env_adapter,
        input_dim,
        action_dim,
        num_episodes=2000,
        seq_len=4,
        batch_size=16,
        mem_usage_callback=None
    ):
    """
    Runs MHENST-DQN for 'num_episodes' episodes.
    Returns:
      rews (list of float) => shape=(episodes,)
      dqn_online, vae, gnn, meta
      vae_rec_list => list(episodes) of average recon error
      q_val_list   => list(episodes) of average Q-value
    """
    # Build modules
    from torch.amp import autocast, GradScaler
    hidden_dim=32
    latent_dim=8

    vae= VAEModule(input_dim, hidden_dim, latent_dim).to(DEVICE)
    vae_opt= optim.Adam(vae.parameters(), lr=1e-3)

    gnn= KnowledgeGraphGNN(node_feature_dim=latent_dim, hidden_dim=16, output_dim=8, node_count=8).to(DEVICE)

    dqn_online= EnhancedTransformerDQN(input_dim=8, action_dim=action_dim,
                                       seq_len=seq_len, num_layers=2, dropout=0.1).to(DEVICE)
    dqn_target= copy.deepcopy(dqn_online).eval().to(DEVICE)
    dqn_opt= optim.Adam(dqn_online.parameters(), lr=1e-4)

    meta= EnhancedMetaCognition(init_epsilon=1.0, min_epsilon=0.01, eps_decay=0.995,
                                init_lr=1e-4, lr_decay=0.999, reward_threshold=0.2)

    replay_buf= SequenceReplayBuffer(capacity=2000, seq_len=seq_len)
    scaler_vae= GradScaler(enabled=(DEVICE.type=='cuda'))
    scaler_dqn= GradScaler(enabled=(DEVICE.type=='cuda'))

    process= psutil.Process(os.getpid())

    rews= []
    vae_rec_list= []
    q_val_list= []

    for ep in range(num_episodes):
        obs, info= env_adapter.reset()
        done= False
        ep_reward= 0.0
        while not done:
            with autocast(device_type='cuda', enabled=(DEVICE.type=='cuda')):
                obs_t= torch.tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0)
                z= vae.encode(obs_t)
                emb= gnn(z)
            seq_input= emb.unsqueeze(1)
            action= meta.select_action(seq_input, dqn_online)

            next_obs, rew, done2, info= env_adapter.step(action)
            ep_reward+= rew
            replay_buf.add_transition(obs, action, rew, next_obs, done2)
            obs= next_obs
            done= done2

        rews.append(ep_reward)
        meta.update_epsilon(ep_reward)

        # Memory usage
        if mem_usage_callback:
            mem_bytes= process.memory_info().rss
            mem_mb= mem_bytes/(1024*1024)
            mem_usage_callback(mem_mb)

        # training
        ep_vae_losses=[]
        ep_qvals=[]
        if len(replay_buf)>= batch_size:
            sequences, idxs, _= replay_buf.sample(batch_size)
            if sequences:
                # VAE step
                states_list= []
                for sq in sequences:
                    for tr in sq:
                        states_list.append(tr.state)
                st_arr= np.array(states_list,dtype=np.float32)

                train_vae_step_collect(vae, st_arr, vae_opt, scaler_vae, ep_vae_losses)

                # DQN step
                train_dqn_step_collect(dqn_online, dqn_target,
                                       vae, gnn,
                                       sequences,
                                       dqn_opt, scaler_dqn,
                                       meta, idxs, replay_buf,
                                       ep_qvals)

        if ep_vae_losses:
            vae_rec_list.append(np.mean(ep_vae_losses))
        else:
            vae_rec_list.append(0.0)

        if ep_qvals:
            q_val_list.append(np.mean(ep_qvals))
        else:
            q_val_list.append(0.0)

        if ep%100==0 and ep>0:
            gc.collect()

    env_adapter.close()
    return rews, dqn_online, vae, gnn, meta, vae_rec_list, q_val_list


###############################################################################
# 6) Minimal Baseline: VanillaDQN
###############################################################################
def run_vanilla_dqn(
    env_adapter,
    input_dim,
    action_dim,
    num_episodes=2000,
    seed=42
):
    """
    A simple MLP-based DQN that logs:
      reward, epsilon, memory usage, vae_error=0, q_value
    Returns dict => { "reward", "epsilon", "memory", "vae_error", "q_value" }
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    epsilon=1.0
    min_eps=0.01
    eps_decay=0.995
    replay= deque(maxlen=2000)

    class MLPQNet(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc1= nn.Linear(state_dim,64)
            self.fc2= nn.Linear(64,64)
            self.fc3= nn.Linear(64,action_dim)
        def forward(self,x):
            x= F.relu(self.fc1(x))
            x= F.relu(self.fc2(x))
            return self.fc3(x)

    qnet= MLPQNet(input_dim, action_dim).to(DEVICE)
    qnet_t= copy.deepcopy(qnet).eval().to(DEVICE)
    opt= optim.Adam(qnet.parameters(), lr=1e-3)

    reward_list= []
    epsilon_list=[]
    memory_list=[]
    vae_error_list=[]
    q_value_list=[]

    process= psutil.Process(os.getpid())

    for ep in range(num_episodes):
        obs,info= env_adapter.reset(seed=seed)
        done= False
        ep_rew= 0.0

        while not done:
            if random.random()< epsilon:
                act= random.randint(0,action_dim-1)
            else:
                obs_t= torch.tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    qvals= qnet(obs_t)
                act= int(torch.argmax(qvals,dim=1).item())
            obs2, rew, done2, info= env_adapter.step(act)
            replay.append((obs,act,rew,obs2,done2))
            obs= obs2
            done= done2
            ep_rew+= rew

        reward_list.append(ep_rew)

        # epsilon
        if epsilon> min_eps:
            epsilon*= eps_decay
        epsilon_list.append(epsilon)

        mem_bytes= process.memory_info().rss
        mem_mb= mem_bytes/(1024*1024)
        memory_list.append(mem_mb)

        vae_error_list.append(0.0)  # no VAE in baseline

        # training
        ep_qbatch=[]
        if len(replay)>=32:
            batch= random.sample(replay,32)
            s_,a_,r_,ns_,d_= [],[],[],[],[]
            for (ss,aa,rr,nss,dd) in batch:
                s_.append(ss)
                a_.append(aa)
                r_.append(rr)
                ns_.append(nss)
                d_.append(dd)
            s_ = torch.tensor(s_,dtype=torch.float32,device=DEVICE)
            a_ = torch.tensor(a_,dtype=torch.long,device=DEVICE)
            r_ = torch.tensor(r_,dtype=torch.float32,device=DEVICE)
            ns_= torch.tensor(ns_,dtype=torch.float32,device=DEVICE)
            d_ = torch.tensor(d_,dtype=torch.float32,device=DEVICE)

            q_pred= qnet(s_)
            q_pred_a= q_pred.gather(1,a_.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next= qnet_t(ns_)
                q_next_max= q_next.max(dim=1)[0]
                tgt= r_ + 0.99*(1.- d_)* q_next_max
            loss= F.smooth_l1_loss(q_pred_a, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_qbatch.append(q_pred.mean().item())

            # soft update
            for p_t,p_o in zip(qnet_t.parameters(), qnet.parameters()):
                p_t.data.copy_(0.01*p_o.data + 0.99*p_t.data)

        if ep_qbatch:
            q_value_list.append(np.mean(ep_qbatch))
        else:
            q_value_list.append(0.0)

        if ep%100==0 and ep>0:
            gc.collect()

    env_adapter.close()
    logs= {
        "reward":   np.array(reward_list),
        "epsilon":  np.array(epsilon_list),
        "memory":   np.array(memory_list),
        "vae_error":np.array(vae_error_list),
        "q_value":  np.array(q_value_list)
    }
    return logs


###############################################################################
# 7) Master script: multi method × multi env × multi seed
###############################################################################
def main():
    # define environment specs => all 2000 episodes
    env_specs = []
    # CartPole
    cp_env= gym.make("CartPole-v1")
    cp_adapter= EnvAdapter(cp_env)
    env_specs.append(("CartPole", cp_adapter, 4, 2, 2000))

    # If minigrid available => 2000 episodes
    if MINIGRID_AVAILABLE:
        mg_raw= DoorKeyEnv(size=8)
        mg_wrap= ImgObsWrapper(mg_raw)
        obs_tmp,_= mg_wrap.reset()
        h,w,c= obs_tmp.shape
        mg_adapter= EnvAdapter(mg_wrap)
        env_specs.append(("MiniGrid-DoorKey", mg_adapter, h*w*c, 7, 2000))

    # define 2 methods (MHENST-DQN, VanillaDQN)
    def run_mhenst_wrapper(env_adapter, input_dim, action_dim, num_episodes, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        mem_usage_list=[]
        def mem_cb(mb):
            mem_usage_list.append(mb)

        rews, dqn, vae, gnn, meta, vae_err_list, q_val_list= run_mhenst_dqn_on_env(
            env_adapter,
            input_dim=input_dim,
            action_dim=action_dim,
            num_episodes=num_episodes,
            seq_len=4,
            batch_size=16,
            mem_usage_callback=mem_cb
        )
        # convert to logs dict
        logs= {}
        logs["reward"]   = np.array(rews)
        logs["epsilon"]  = np.array(meta.epsilon_history)
        logs["memory"]   = np.array(mem_usage_list)
        logs["vae_error"]= np.array(vae_err_list)
        logs["q_value"]  = np.array(q_val_list)
        return logs

    def run_vanilla_wrapper(env_adapter, input_dim, action_dim, num_episodes, seed):
        logs= run_vanilla_dqn(
            env_adapter,
            input_dim= input_dim,
            action_dim= action_dim,
            num_episodes=num_episodes,
            seed= seed
        )
        return logs

    methods= {
        "MHENST-DQN": run_mhenst_wrapper,
        "VanillaDQN": run_vanilla_wrapper
    }

    seed_list= [42, 123]
    # We'll store => all_results[env][method][metric] => shape=(num_seeds, episodes)
    all_results= defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    out_pdf= "comparison_results.pdf"
    pdf= PdfPages(out_pdf)

    for (env_name, adapter, obs_dim, act_dim, max_eps) in env_specs:
        for method_name, method_func in methods.items():
            print(f"\n=== {method_name} on {env_name} for {max_eps} episodes ===")
            for sd in seed_list:
                logs= method_func(adapter, obs_dim, act_dim, max_eps, sd)
                # logs => keys=[reward, epsilon, memory, vae_error, q_value]
                for metric, arr in logs.items():
                    all_results[env_name][method_name][metric].append(arr)

    # unify shape => pad
    for env_name, method_dict in all_results.items():
        for method_name, metric_dict in method_dict.items():
            for metric, list_of_arr in metric_dict.items():
                max_len= max(len(a) for a in list_of_arr)
                new_list=[]
                for arr in list_of_arr:
                    if len(arr)< max_len:
                        pad_sz= max_len- len(arr)
                        arr= np.concatenate([arr, np.full(pad_sz, arr[-1])])
                    new_list.append(arr)
                metric_dict[metric]= np.array(new_list)  # shape=(num_seeds, max_len)

    # Plot multi-subplot => [reward, epsilon, memory, vae_error, q_value]
    METRICS_ORDER= ["reward","epsilon","memory","vae_error","q_value"]
    METRICS_LABEL={
        "reward":"Reward",
        "epsilon":"Epsilon",
        "memory":"Memory(MB)",
        "vae_error":"VAE Error",
        "q_value":"Average Q-value"
    }

    for env_name, method_dict in all_results.items():
        fig, axes= plt.subplots(nrows=5, ncols=1, figsize=(7,18))
        fig.suptitle(f"{env_name} - Detailed metrics (N={len(seed_list)} seeds, each 2000 eps)")

        for i, metric in enumerate(METRICS_ORDER):
            ax= axes[i]
            for method_name, metric_data in method_dict.items():
                data_array= metric_data[metric]  # shape=(num_seeds, episodes)
                mean_val= data_array.mean(axis=0)
                std_val = data_array.std(axis=0)
                x_axis  = np.arange(data_array.shape[1])
                ax.plot(x_axis, mean_val, label=method_name)
                ax.fill_between(x_axis, mean_val-std_val, mean_val+std_val, alpha=0.2)
            ax.set_title(METRICS_LABEL[metric])
            ax.set_xlabel("Episode")

        axes[0].legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    print(f"\nAll done! See {out_pdf} for multi-subplot metrics.\n")

    # if exactly 2 methods => ttest final 50 episodes of reward
    if len(methods)==2:
        method_list= list(methods.keys())
        m1,m2= method_list[0], method_list[1]
        for env_name, method_dict in all_results.items():
            arr1= method_dict[m1]["reward"]  # shape=(num_seeds, episodes)
            arr2= method_dict[m2]["reward"]
            final_1= arr1[:, -50:].mean(axis=1)
            final_2= arr2[:, -50:].mean(axis=1)
            t_stat,p_val= ttest_ind(final_1, final_2, equal_var=False)
            print(f"[T-test] {env_name}, {m1} vs {m2} => p={p_val:.4g}, t={t_stat:.3f}")


if __name__=="__main__":
    main()
