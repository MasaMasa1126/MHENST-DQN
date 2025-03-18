# ========== MHENST-DQN.py ==========
"""
MHENST-DQN (VAE + GNN + Transformer + メタ認知) を複数環境で実行し、
以下をPDFにまとめる:
  - 報酬曲線
  - 探索率(Epsilon)
  - メモリ使用量
  - VAEの再構成誤差(MSE)
  - DQNの平均Q値
さらに、tqdm によりエピソード学習の進捗バーを表示。
"""

import os
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from collections import deque, namedtuple
import copy

# 進捗バー
from tqdm import tqdm

# Mixed Precision
from torch.cuda.amp import autocast, GradScaler

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from minigrid.envs import DoorKeyEnv

# 独自環境
from custom_envs import SequenceMemoryEnv, MultiModalPuzzleEnv

# torch_geometric (GNN)
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (device.type=='cuda')
print(f"[INFO] Using device={device}, USE_AMP={USE_AMP}")

Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))


# ======================================================================
# 1. EnvAdapter
# ======================================================================
class EnvAdapter:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def _process_dict_obs(self, obs):
        # MultiModalPuzzleEnv 用
        out_vars = obs["vars"]            # shape(3,)
        image = obs["image"]              # shape(16,16,3)
        text = obs["text_token"]          # int
        flat_img = image.flatten()         # (768,)
        text_oh = np.zeros(4,dtype=np.float32)
        text_oh[text] = 1.0
        return np.concatenate([out_vars, flat_img, text_oh], axis=0)

    def _process_image_obs(self, obs):
        # MiniGrid DoorKey => shape(7,7,3)等
        obs = obs.astype(np.float32, copy=False)
        return obs.flatten()

    def preprocess_observation(self, obs):
        if isinstance(obs, dict):
            return self._process_dict_obs(obs)
        if hasattr(obs, 'shape') and len(obs.shape)>1:
            return self._process_image_obs(obs)
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)
        return obs

    def reset(self, seed=None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        return self.preprocess_observation(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs_p = self.preprocess_observation(obs)
        done = done or truncated
        return obs_p, reward, done, info

    def close(self):
        self.env.close()


# ======================================================================
# 2. ReplayBuffer, VAE, GNN, Transformer, メタ認知
# ======================================================================
Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class SequenceReplayBuffer:
    def __init__(self, capacity=5000, seq_len=4, alpha=0.6):
        self.capacity= capacity
        self.seq_len= seq_len
        self.buffer= deque(maxlen=capacity)
        self.episode_buffer= []
        self.alpha= alpha
        self.priorities= np.ones((capacity,), dtype=np.float32)
        self.position=0
        self.is_full=False

    def add_transition(self, state, action, reward, next_state, done):
        self.episode_buffer.append(Transition(state,action,reward,next_state,done))
        if done:
            self._process_episode()
            self.episode_buffer=[]

    def _process_episode(self):
        ep_len= len(self.episode_buffer)
        if ep_len==0:
            return
        if ep_len< self.seq_len:
            pad= [self.episode_buffer[0]]*(self.seq_len-ep_len)+ self.episode_buffer
            self._add_sequence(pad[-self.seq_len:])
        else:
            stride= max(1, min(3, ep_len//4))
            for i in range(0, ep_len- self.seq_len+1, stride):
                seq= self.episode_buffer[i:i+self.seq_len]
                self._add_sequence(seq)

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
                self.is_full=True

    def sample(self, batch_size, beta=0.4):
        bsz= min(batch_size, len(self.buffer))
        if bsz==0:
            return None,None,None
        valid_sz= len(self.buffer)
        p= self.priorities[:valid_sz]** self.alpha
        p/= p.sum()
        idxs= np.random.choice(valid_sz, bsz, replace=False, p=p)
        weights= (valid_sz*p[idxs])**(-beta)
        weights/= weights.max()
        samples= [self.buffer[i] for i in idxs]
        return samples, idxs, weights

    def update_priorities(self, idxs, td_errs):
        for idx,err in zip(idxs,td_errs):
            if idx< len(self.priorities):
                self.priorities[idx]= abs(err)+1e-6

    def __len__(self):
        return len(self.buffer)


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.fc1= nn.Linear(in_dim,hid_dim)
        self.fc2= nn.Linear(hid_dim,hid_dim)
        self.ln= nn.LayerNorm(hid_dim)
    def forward(self,x):
        h= F.relu(self.fc1(x))
        h= self.fc2(h)
        return self.ln(h)

class VariationalEncoder(nn.Module):
    def __init__(self,in_dim,hid_dim,lat_dim):
        super().__init__()
        self.enc= EncoderBlock(in_dim,hid_dim)
        self.fc_mu= nn.Linear(hid_dim, lat_dim)
        self.fc_logv= nn.Linear(hid_dim, lat_dim)
    def forward(self,x):
        h= self.enc(x)
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
        z,_,_ = self.encoder(x)
        return z

    def compute_loss(self,x,xr,mu,logv):
        recon= F.mse_loss(xr,x,reduction='sum')
        kl= -0.5* torch.sum(1+ logv- mu.pow(2)- logv.exp())
        return recon+ 0.1* kl


class KnowledgeGraphGNN(nn.Module):
    def __init__(self,node_feature_dim, hidden_dim=16, output_dim=8, node_count=8):
        super().__init__()
        self.num_nodes= node_count
        self.node_features= nn.Parameter(torch.randn(node_count,node_feature_dim)*0.1, requires_grad=True)
        self.conv1= GCNConv(node_feature_dim, hidden_dim)
        self.conv2= GCNConv(hidden_dim, output_dim)

        edges=[]
        for i in range(node_count):
            for j in range(node_count):
                if i!= j:
                    edges.append([i,j])
        self.edge_index= torch.tensor(edges,dtype=torch.long).t().contiguous()
        self.edge_attr= nn.Parameter(torch.ones(self.edge_index.size(1),1), requires_grad=True)
        self.symbolic_rules=[]

    def update_graph_with_latent(self, latent_vec):
        bsz= latent_vec.size(0)
        base= self.node_features.unsqueeze(0).expand(bsz,-1,-1)
        w= torch.softmax(latent_vec[:,:self.num_nodes], dim=1).unsqueeze(-1)
        return base+ 0.1* w

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
        return torch.stack(outs,dim=0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq=100):
        super().__init__()
        pe= torch.zeros(max_seq,d_model)
        position= torch.arange(0,max_seq,dtype=torch.float).unsqueeze(1)
        div= torch.exp(torch.arange(0,d_model,2).float()*(-np.log(10000.0)/d_model))
        pe[:,0::2]= torch.sin(position*div)
        pe[:,1::2]= torch.cos(position*div)
        pe= pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        sl= x.size(1)
        return x+ self.pe[:,:sl,:]

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.attn= nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout)
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
        self.input_dim= input_dim
        self.action_dim= action_dim
        self.seq_len= seq_len
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
            feat_t= x[:,t,:]
            adv= self.adv_stream(feat_t)
            val= self.val_stream(feat_t)
            q_t= val+ adv- adv.mean(dim=1, keepdim=True)
            out_list.append(q_t.unsqueeze(1))
        q_all= torch.cat(out_list,dim=1)
        return q_all


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
                self.epsilon= min(self.epsilon*1.01,0.8)
            else:
                self.epsilon= max(self.min_epsilon, self.epsilon*self.eps_decay)
        else:
            self.epsilon= max(self.min_epsilon, self.epsilon*self.eps_decay)
        self.epsilon_history.append(self.epsilon)
        return self.epsilon

    def update_learning_rate(self, optimizer, loss_val=None):
        self.step_count+=1
        if self.step_count%50==0:
            new_lr= max(self.min_lr, self.lr* self.lr_decay)
            if loss_val is not None and loss_val>1e3:
                new_lr= max(self.min_lr, new_lr*0.9)
            for pg in optimizer.param_groups:
                pg['lr']= new_lr
            self.lr= new_lr
            self.lr_history.append(new_lr)

    def select_action(self, seq_input, dqn):
        if random.random()< self.epsilon:
            return random.randint(0,dqn.action_dim-1)
        else:
            with torch.no_grad():
                q_all= dqn(seq_input)
                q_last= q_all[:,-1,:]
                act= int(torch.argmax(q_last,dim=1).item())
            return act


# ======================================================================
# 3. 学習パイプラインに tqdm を導入
# ======================================================================
def run_mhenst_dqn_on_env(env_adapter,
                          input_dim,
                          action_dim,
                          num_episodes=300,
                          seq_len=4,
                          batch_size=16,
                          mem_usage_callback=None):

    hidden_dim=32
    latent_dim=8
    vae= VAEModule(input_dim, hidden_dim, latent_dim).to(device)
    vae_opt= optim.Adam(vae.parameters(), lr=1e-3)

    gnn= KnowledgeGraphGNN(node_feature_dim=latent_dim, hidden_dim=16, output_dim=8, node_count=8).to(device)

    dqn_online= EnhancedTransformerDQN(input_dim=8, action_dim=action_dim,
                                       seq_len=seq_len, num_layers=2, dropout=0.1).to(device)
    dqn_target= copy.deepcopy(dqn_online).eval().to(device)
    dqn_opt= optim.Adam(dqn_online.parameters(), lr=1e-4)

    meta= EnhancedMetaCognition(init_epsilon=1.0, min_epsilon=0.01, eps_decay=0.995,
                                init_lr=1e-4, lr_decay=0.999, reward_threshold=0.2)

    buffer= SequenceReplayBuffer(capacity=2000, seq_len=seq_len)

    scaler_vae= GradScaler(enabled=USE_AMP)
    scaler_dqn= GradScaler(enabled=USE_AMP)

    all_rewards=[]
    process = psutil.Process(os.getpid())

    vae_rec_loss_history = []
    avg_q_history = []

    # --- tqdm でエピソードループ ---
    for ep in tqdm(range(num_episodes), desc=f"[{env_adapter.env.unwrapped.__class__.__name__}] Episodes", leave=True):
        obs, info= env_adapter.reset()
        done=False
        ep_reward=0.0

        while not done:
            with autocast(enabled=USE_AMP):
                obs_t= torch.tensor(obs,dtype=torch.float32, device=device)
                if obs_t.ndim==1:
                    obs_t= obs_t.unsqueeze(0)
                z= vae.encode(obs_t)
                emb= gnn(z)
            seq_input= emb.unsqueeze(1)
            action= meta.select_action(seq_input, dqn_online)

            next_obs, reward, done2, info= env_adapter.step(action)
            ep_reward+= reward
            buffer.add_transition(obs, action, reward, next_obs, done2)
            obs= next_obs
            done= done2

        all_rewards.append(ep_reward)
        meta.update_epsilon(ep_reward)

        # メモリ使用量
        if mem_usage_callback:
            mem_bytes = process.memory_info().rss
            mem_mb = mem_bytes / (1024*1024)
            mem_usage_callback(mem_mb)

        # VAE再構成誤差 & Q値の集計
        ep_vae_losses = []
        ep_q_values = []

        if len(buffer)>= batch_size:
            sequences, idxs, _= buffer.sample(batch_size)
            if sequences:
                # VAE学習
                states_list=[]
                for seq_obj in sequences:
                    for tr in seq_obj:
                        states_list.append(tr.state)
                st_arr= np.array(states_list,dtype=np.float32)

                _ = train_vae_step_collect(vae, st_arr, vae_opt, scaler_vae, ep_vae_losses)

                # DQN学習
                train_dqn_step_collect(dqn_online, dqn_target,
                                       vae, gnn,
                                       sequences,
                                       dqn_opt, scaler_dqn,
                                       meta, idxs, buffer,
                                       ep_q_values)

        avg_rec_loss = np.mean(ep_vae_losses) if len(ep_vae_losses)>0 else 0.0
        vae_rec_loss_history.append(avg_rec_loss)

        avg_q_val = np.mean(ep_q_values) if len(ep_q_values)>0 else 0.0
        avg_q_history.append(avg_q_val)

    env_adapter.close()

    return all_rewards, dqn_online, vae, gnn, meta, vae_rec_loss_history, avg_q_history


# 追加の学習補助関数
def train_vae_step_collect(vae, states_array, optimizer, scaler, ep_vae_losses):
    vae.train()
    if not isinstance(states_array, torch.Tensor):
        states_array= torch.tensor(states_array,dtype=torch.float32,device=device)
    optimizer.zero_grad()
    with autocast(enabled=USE_AMP):
        xr,z,mu,logv= vae(states_array)
        loss= vae.compute_loss(states_array, xr, mu, logv)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        recon_mse = F.mse_loss(xr, states_array, reduction='mean').item()
        ep_vae_losses.append(recon_mse)
    return loss.item()


def train_dqn_step_collect(dqn_online, dqn_target,
                           vae, gnn,
                           sequences,
                           dqn_opt, scaler_dqn,
                           meta, idxs, buffer,
                           ep_q_values):
    td_errs=[]
    for seq in sequences:
        s_buf, a_buf, r_buf, ns_buf, d_buf= [],[],[],[],[]
        for tr in seq:
            s_buf.append(tr.state)
            a_buf.append(tr.action)
            r_buf.append(tr.reward)
            ns_buf.append(tr.next_state)
            d_buf.append(tr.done)
        s_buf= np.array(s_buf,dtype=np.float32)
        a_buf= np.array(a_buf,dtype=np.int64)
        r_buf= np.array(r_buf,dtype=np.float32)
        ns_buf= np.array(ns_buf,dtype=np.float32)
        d_buf= np.array(d_buf,dtype=np.float32)

        s_t= torch.tensor(s_buf,device=device)
        a_t= torch.tensor(a_buf,device=device)
        r_t= torch.tensor(r_buf,device=device)
        ns_t= torch.tensor(ns_buf,device=device)
        d_t= torch.tensor(d_buf,device=device)

        with torch.no_grad():
            z_s= vae.encode(s_t)
            z_ns= vae.encode(ns_t)
            emb_s= gnn(z_s)
            emb_ns= gnn(z_ns)
        emb_s= emb_s.unsqueeze(0)
        emb_ns= emb_ns.unsqueeze(0)

        with autocast(enabled=USE_AMP):
            q_all= dqn_online(emb_s)
            q_last= q_all[:,-1,:]
            act_last= a_t[-1]
            rew_last= r_t[-1]
            done_last= d_t[-1]

            q_pred= q_last[0, act_last]
            q_next= dqn_target(emb_ns)
            q_last_ns= q_next[:,-1,:]
            q_next_max= q_last_ns.max(dim=1)[0]
            gamma=0.99
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
            avgQ = q_last.mean().item()
            ep_q_values.append(avgQ)

    buffer.update_priorities(idxs, td_errs)
    # ソフト更新
    for p_t, p_o in zip(dqn_target.parameters(), dqn_online.parameters()):
        p_t.data.copy_(0.01* p_o.data + 0.99* p_t.data)


# ======================================================================
# 4. GNN 可視化
# ======================================================================
def visualize_knowledge_graph(gnn):
    num_nodes= gnn.num_nodes
    edge_idx= gnn.edge_index.detach().cpu().numpy()
    edge_attr= gnn.edge_attr.detach().cpu().numpy().flatten()

    G= nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(edge_idx.shape[1]):
        src= edge_idx[0,i]
        dst= edge_idx[1,i]
        w= float(edge_attr[i])
        G.add_edge(src,dst, weight=w)

    pos= nx.spring_layout(G)
    plt.figure()
    edges= G.edges()
    w_list= [G[u][v]['weight'] for u,v in edges]
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)
    labels= {n: f"Node{n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    nx.draw_networkx_edges(G, pos, width=[max(0.1, w*2) for w in w_list], alpha=0.7)
    plt.title("Learned Knowledge Graph (GNN)")
    plt.axis("off")


# ======================================================================
# 5. メイン実行 (PDF出力)
# ======================================================================
def main():
    # 環境リスト
    envs = []
    cartpole_raw= gym.make("CartPole-v1")
    cartpole_env= EnvAdapter(cartpole_raw)
    envs.append(("CartPole", cartpole_env, 4, 2))

    minigrid_raw= DoorKeyEnv(size=8)
    minigrid_wrapped= ImgObsWrapper(minigrid_raw)
    obs_tmp, _= minigrid_wrapped.reset()
    h,w,c= obs_tmp.shape
    input_dim_mg= h*w*c
    minigrid_env= EnvAdapter(minigrid_wrapped)
    envs.append(("MiniGrid-DoorKey", minigrid_env, input_dim_mg, 7))

    memory_raw= SequenceMemoryEnv(seq_length=5)
    memory_env= EnvAdapter(memory_raw)
    envs.append(("SequenceMemory", memory_env, 1, 2))

    puzzle_raw= MultiModalPuzzleEnv()
    puzzle_env= EnvAdapter(puzzle_raw)
    obs_tmp2, _= puzzle_env.reset()
    envs.append(("MultiModalPuzzle", puzzle_env, obs_tmp2.size, 6))

    with PdfPages("MHENST_results.pdf") as pdf:
        for (env_name, adapter, obs_dim, act_dim) in envs:
            print(f"\n=== Training MHENST-DQN on {env_name} ===")

            mem_usage_list= []
            def mem_callback(mb):
                mem_usage_list.append(mb)

            # run_mhenst_dqn_on_env で tqdm 付きループ実行
            rews, dqn, vae, gnn, meta, vae_rec_loss_hist, avg_q_hist = run_mhenst_dqn_on_env(
                adapter,
                input_dim=obs_dim,
                action_dim=act_dim,
                num_episodes=5000,  # たとえば
                seq_len=4,
                batch_size=16,
                mem_usage_callback=mem_callback
            )

            # 報酬
            plt.figure()
            plt.plot(rews, label=env_name)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"Reward on {env_name}")
            plt.legend()
            pdf.savefig()
            plt.close()

            # Epsilon
            plt.figure()
            plt.plot(meta.epsilon_history, label=f"{env_name} Eps")
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            plt.title(f"Epsilon Decay on {env_name}")
            plt.legend()
            pdf.savefig()
            plt.close()

            # メモリ使用量
            plt.figure()
            plt.plot(mem_usage_list, label=f"{env_name} MemUsage(MB)")
            plt.xlabel("Episode")
            plt.ylabel("Memory Usage (MB)")
            plt.title(f"Memory usage over episodes: {env_name}")
            plt.legend()
            pdf.savefig()
            plt.close()

            # VAE再構成誤差
            plt.figure()
            plt.plot(vae_rec_loss_hist, label=f"{env_name} VAE RecLoss")
            plt.xlabel("Episode")
            plt.ylabel("MSE (Reconstruction)")
            plt.title(f"VAE reconstruction error: {env_name}")
            plt.legend()
            pdf.savefig()
            plt.close()

            # 平均Q値
            plt.figure()
            plt.plot(avg_q_hist, label=f"{env_name} AvgQ")
            plt.xlabel("Episode")
            plt.ylabel("Avg Q-Value")
            plt.title(f"Average Q-value: {env_name}")
            plt.legend()
            pdf.savefig()
            plt.close()

            # GNN可視化 (DoorKey, MultiModalPuzzle)
            if env_name in ["MiniGrid-DoorKey","MultiModalPuzzle"]:
                visualize_knowledge_graph(gnn)
                pdf.savefig()
                plt.close()

        print("\n=== All experiments done. ===")
        print("Results saved in MHENST_results.pdf")


if __name__=="__main__":
    main()
