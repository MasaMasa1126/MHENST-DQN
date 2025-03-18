# ========== custom_envs.py ==========
"""
### 1. `custom_envs.py`の追加
`MultiModalPuzzleEnv`は、神経シンボリックアプローチの優位性を示すのに最適な環境です:
- マルチモーダル入力: 数値変数、画像、テキストトークンの3つのモダリティを組み合わせ
- シンボリック条件: `varA + varB == 3` かつ `varC == 1`という論理条件
- 視覚認識要素: 画像の平均値に基づく条件
- 言語的指示理解: テキストトークンに基づいた適切な行動選択

### 2. マルチ環境対応の完成度
本ファイルや MHENST-DQN.pyでは、複数の異なる環境（CartPole、MiniGrid、SequenceMemory、LogicPuzzleなど）を
一貫したインターフェースで扱っており、以下の点が優れています:
- `EnvAdapter`による観測の自動的な前処理
- 各環境用のディメンション設定の適切な管理
- 視覚的なフィードバック（報酬グラフ、εの減衰、知識グラフ）の統一的な提供

### 3. 要素
この実装は論文化に最適な以下の要素を含んでいます:
1. マルチモーダル性: 画像、数値、離散トークンを統合処理する能力
2. 抽象的推論: 論理パズル環境での推論能力の実証
3. 一般化能力: 多様な環境に同一アーキテクチャで対応
4. 可解釈性: 知識グラフやその他の可視化による内部表現の解釈
5. 効率性: AMP対応と最適化された実装

"""

import gymnasium as gym
import numpy as np


# ======================================================================
# 1. SequenceMemoryEnv
# ======================================================================
class SequenceMemoryEnv(gym.Env):
    """
    シンプルなメモリタスク:
      - seq_lengthビット列 self.sequence
      - 各ステップ i で観測=[i],行動=0/1
      - action==sequence[i] => +1, else -1
      - i>=seq_length で終了
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, seq_length=5):
        super().__init__()
        self.seq_length = seq_length
        self.observation_space = gym.spaces.Box(
            low=0, high=float(seq_length), shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)

        self.sequence = None
        self.current_step = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = np.random.randint(0,2,size=(self.seq_length,), dtype=int)
        self.current_step = 0
        obs = np.array([self.current_step], dtype=np.float32)
        return obs, {}

    def step(self, action):
        correct = (action == self.sequence[self.current_step])
        reward = 1.0 if correct else -1.0
        self.current_step +=1
        done= (self.current_step>= self.seq_length)
        truncated= False

        if not done:
            obs= np.array([self.current_step], dtype=np.float32)
        else:
            obs= np.array([self.current_step], dtype=np.float32)
        return obs, reward, done, truncated, {}


# ======================================================================
# 2. マルチモーダル論理パズル (MultiModalPuzzleEnv)
# ======================================================================
class MultiModalPuzzleEnv(gym.Env):
    """
    マルチモーダルかつ神経シンボリックを想定した論理パズル:
      - vars: varA, varB in [0..3], varC in [0..1]
      - image: 16x16x3, float(0..1)
      - text_token: 離散(0..3)
      行動(6種類):
        0: varA+=1, 1: varA-=1, 2: varB+=1, 3: varB-=1,
        4: varC=1-varC, 5: no-op
      報酬例:
        - 基本 -0.1
        - varA+varB==3 & varC==1 => +1.0
        - 画像平均>0.5 => +0.5
        - テキストトークンに応じた行動 => +0.2
    """
    metadata = {"render_modes":["human"]}

    def __init__(self):
        super().__init__()

        # 数値変数: Box([0,0,0],[3,3,1])
        self.vars_space = gym.spaces.Box(
            low=np.array([0,0,0],dtype=np.float32),
            high=np.array([3,3,1],dtype=np.float32),
            shape=(3,), dtype=np.float32
        )
        # 画像: 16x16x3
        self.image_space= gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(16,16,3), dtype=np.float32
        )
        # テキストトークン: Discrete(4)
        self.text_space= gym.spaces.Discrete(4)

        # 観測は dict
        self.observation_space= gym.spaces.Dict({
            "vars": self.vars_space,
            "image": self.image_space,
            "text_token": self.text_space,
        })

        # 行動
        self.action_space= gym.spaces.Discrete(6)

        self.max_steps=10
        self.steps=0
        self.varA=0
        self.varB=0
        self.varC=0
        self.image=None
        self.text_token=0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps=0
        # vars
        self.varA= np.random.randint(0,4)
        self.varB= np.random.randint(0,4)
        self.varC= np.random.randint(0,2)
        # 画像
        self.image= np.random.rand(16,16,3).astype(np.float32)
        # テキストトークン
        self.text_token= np.random.randint(0,4)

        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "vars": np.array([self.varA,self.varB,self.varC],dtype=np.float32),
            "image": self.image,
            "text_token": self.text_token,
        }

    def step(self, action):
        # 行動適用
        if action==0:
            self.varA= min(3, self.varA+1)
        elif action==1:
            self.varA= max(0, self.varA-1)
        elif action==2:
            self.varB= min(3, self.varB+1)
        elif action==3:
            self.varB= max(0, self.varB-1)
        elif action==4:
            self.varC= 1- self.varC
        elif action==5:
            pass

        # 報酬
        reward= -0.1
        # (1) 論理条件
        if (self.varA+ self.varB==3) and (self.varC==1):
            reward+=1.0
        # (2) 画像平均
        mean_pix= float(self.image.mean())
        if mean_pix>0.5:
            reward+=0.5
        # (3) テキストトークン
        #   0:"INCREMENT_A"(action=0)
        #   1:"DECREMENT_B"(action=3)
        #   2:"TOGGLE_C"  (action=4)
        #   3:"NO_OP"     (action=5)
        if self.text_token==0 and action==0:
            reward+=0.2
        elif self.text_token==1 and action==3:
            reward+=0.2
        elif self.text_token==2 and action==4:
            reward+=0.2
        elif self.text_token==3 and action==5:
            reward+=0.2

        self.steps+=1
        done= (self.steps>=self.max_steps)
        truncated=False

        # ランダムに画像変更(10%)
        if np.random.rand()<0.1:
            self.image= np.random.rand(16,16,3).astype(np.float32)
        # テキストトークン変更(5%)
        if np.random.rand()<0.05:
            self.text_token= np.random.randint(0,4)

        obs= self._get_obs()
        return obs, reward, done, truncated, {}

    def render(self):
        print(f"Step={self.steps}, A={self.varA}, B={self.varB}, C={self.varC}, text={self.text_token}, meanPix={self.image.mean():.2f}")
