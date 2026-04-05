# ==========================================================
# 🏆 【V18.2・中トロ最終決戦】8xGPU 3周ループ＆お持ち帰り完全版
# 🇯🇵 パッチン流ギッチギチ収納術（608次元 ＋ リバーシブルV8エンジン）
# ==========================================================
set -e

echo "👷 決勝戦の朝礼開始！ ウサギ小屋の叡智・V18.2（608次元）出撃！！"

# 1. インフラ＆教科書準備
apt-get update && apt-get install -y git psmisc zip
pip install torch>=2.5.0 --upgrade --index-url https://download.pytorch.org/whl/cu124
pip install sentencepiece numpy huggingface-hub datasets tqdm zstandard requests
mkdir -p /workspace/parameter-golf && cd /workspace/parameter-golf
[ ! -d "data" ] && git clone https://github.com/openai/parameter-golf.git .
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# 2. V18.2パッチ（運命の608次元 ＋ 表裏一体リバーシブル機構）
cat << 'EOF_PATCH' > apply_patch_v18_final.py
import re
with open("train_gpt.py", "r") as f: lines = f.readlines()
new_lines = []
bigram_added = False
bigram_logic = """
import torch
import torch.nn.functional as F_fast
class BigramHashEmbedding(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.vocab_size = base.weight.shape[0]
    @property
    def weight(self): return self.base.weight
    def forward(self, x):
        direct_emb = self.base(x)
        x_prev = torch.zeros_like(x); x_prev[:, 1:] = x[:, :-1]
        hash_idx = (x_prev * 31337 + x) % self.vocab_size
        return direct_emb + 0.3 * F_fast.embedding(hash_idx, self.base.weight)
"""
for line in lines:
    new_lines.append(line)
    if "from __future__ import annotations" in line and not bigram_added:
        new_lines.append(bigram_logic)
        bigram_added = True
    if "model_dim =" in line or "n_embd =" in line:
        if "=" in line and ("512" in line or "int(" in line or "448" in line or "480" in line or "768" in line or "640" in line or "608" in line):
            var_name = line.split("=")[0].strip()
            new_lines[-1] = f"    {var_name} = 608 # 統括命令：ツナ缶限界ギッチギチの608次元\n"

code = "".join(new_lines)
code = re.sub(r'ctx = torch\.amp\.autocast\(device_type=device_type, dtype=ptdtype\)', r'ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)', code)
code = re.sub(r'F\.gelu\((.*?)\)', r'torch.pow(torch.nn.functional.leaky_relu(\1, negative_slope=0.5), 2.0)', code)
code = code.replace("self.tok_emb = nn.Embedding(vocab_size, model_dim)", "self.tok_emb_base = nn.Embedding(vocab_size, model_dim)\n        self.tok_emb = BigramHashEmbedding(self.tok_emb_base)")
xsa_code = "        x_p = torch.zeros_like(x); x_p[:, 1:] = x[:, :-1]\n        qkv = self.c_attn(0.8 * x + 0.2 * x_p)"
code = code.replace("qkv = self.c_attn(x)", xsa_code)
code = code.replace("self.lm_head.weight = self.tok_emb.weight", "self.lm_head.weight = self.tok_emb_base.weight")
code = code.replace("self.tok_emb.weight = self.lm_head.weight", "self.tok_emb_base.weight = self.lm_head.weight")
code = code.replace("nn.init.normal_(self.tok_emb.weight", "nn.init.normal_(self.tok_emb_base.weight")
code = code.replace("[base_model.tok_emb.weight]", "[base_model.tok_emb_base.weight]")
code = re.sub(r'F\.linear\((.*?), self\.tok_emb\.weight\)', r'F.linear(\1, self.tok_emb_base.weight)', code)

# 👇 パッチン流リバーシブル機構
rev_logic = """
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'h'):
            n_l = len(self.transformer.h)
            for idx in range(n_l // 2):
                self.transformer.h[n_l - 1 - idx] = self.transformer.h[idx]
"""
code = code.replace("self.apply(self._init_weights)", rev_logic + "\n        self.apply(self._init_weights)")

with open("train_gpt_v18_final.py", "w") as f: f.write(code)
EOF_PATCH
python3 apply_patch_v18_final.py

# 3. 🏁 8基連動 × 3回連続の爆走ループ (本番の13000歩仕様！)
set +e
for i in 1 2 3; do
    echo "🔥 第 ${i} レース出走！ (SEED=${i}) 頼むぞ中トロエンジン！！"
    SEED=${i} TORCH_COMPILE_DISABLE=1 RUN_ID=v18_final_run${i} NUM_LAYERS=10 MODEL_DIM=608 WARMDOWN_ITERS=3000 \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 ITERATIONS=13000 \
    TRAIN_BATCH_TOKENS=32768 TRAIN_SEQ_LEN=1024 VAL_LOSS_EVERY=250 VAL_BATCH_SIZE=8192 \
    torchrun --standalone --nproc_per_node=8 train_gpt_v18_final.py 2>&1 | tee run_${i}.log
done
set -e

# 4. 📦 お土産のダンボール詰め
echo "📦 3回の走行完了！激太りしたモデルをzipに梱包します..."
zip submission_v18_final.zip run_1.log run_2.log run_3.log train_gpt_v18_final.py
echo "================================================================"
echo "🚨 統括！！学習がすべて完了し、運命の submission_v18_final.zip が完成しました！"
echo "🚨 ダウンロードして、16.0MBの壁を越えていないか確認してください！！"
echo "================================================================"

# 5. 💣 手動自爆ギミック
read -p "ダウンロードは完了しましたか？ (Enterを押すと現場を吹き飛ばします): " dummy_var
echo "💣 パッチン流・大和魂の勝利を信じて！爆破！！"
kill -9 -1
