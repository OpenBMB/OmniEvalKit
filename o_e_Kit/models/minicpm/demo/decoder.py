import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('inf')):
    logits = logits.clone()

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个超过top_p的
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value

    return logits

class StreamDecoder:
    def __init__(self, llm, tokenizer, special_token_ids=None):
        self.m = llm
        self.tokenizer = tokenizer
        self.listen_id = self.tokenizer.eos_token_id
        
        # 设置特殊 token IDs
        self.chunk_eos_id = self.tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        self.chunk_tts_eos_id = self.tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        self.turn_eos_id = self.tokenizer.convert_tokens_to_ids("<|turn_eos|>")
        self.speak_id = self.tokenizer.convert_tokens_to_ids("<|speak|>")
        
        # 如果提供了特殊 token IDs，使用它们
        if special_token_ids:
            self.special_token_ids = special_token_ids
        else:
            self.special_token_ids = []
        
        self.cache = None
        self.context = ''
        self.generated_tokens = []  # 跟踪已生成的 tokens
        self.reset()
        self.embeds = None
        self.system_embeds = None

    def sliding_embeds(self):
        # tmp = system_embeds
        # tmp +-》 embeds 后 5s
        # reset
        # feed 
        pass
    
    def reset(self):
        self.context = ''
        self.cache = None
        self.generated_tokens = []  # 重置已生成的 tokens

    def get_context(self):
        return self.context

    def embed_token(self, tid):
        if isinstance(tid, int):
            tid = torch.tensor([tid], device=self.m.device)
        return self.m.model.embed_tokens(tid)

    @torch.no_grad()
    def feed(self, embeds: torch.Tensor, return_logits: bool = False):
        """
        embeds : [L, H]   —— 一次送进模型的新 embedding 序列
        """
        L = embeds.size(0)
        device = embeds.device

        # -------- 1. 计算 position_ids --------
        past_len = 0
        if self.cache is not None:
            past_len = self.cache[0][0].shape[2]        # 取第 1 层 K 的 seq_len
        pos_ids = torch.arange(past_len, past_len + L, device=device).unsqueeze(0)
        out = self.m(
            inputs_embeds = embeds.unsqueeze(0),   # [1, L, H]
            position_ids  = pos_ids,
            past_key_values = self.cache,
            return_dict = True,
            output_hidden_states=True,
        )
        self.cache = out.past_key_values          # 更新 KV-cache

        # -------- 3. Optional 返回 logits --------
        if return_logits:
            logits = self.m.lm_head(out.hidden_states[-1])[:, -1]
            return logits, out.hidden_states[-1]

    @torch.no_grad()
    def decode(
        self,
        logits,
        temperature=0.7,
        mode="sampling",
        top_k=20,
        top_p=0.8,
        listen_top_k=None,
        listen_prob_scale=1.0,
        text_repetition_penalty=1.05,
        text_repetition_window_size=512,
        debug_print_top5=False,
    ):
        """
        mode: "sampling" or "greedy"
        listen_top_k: 强制要求 listen_id 至少在 top-k 中才保留
        listen_prob_scale: 对 listen_id 概率单独乘一个权重（<1 代表降低，>1 代表提升）
        text_repetition_penalty: 重复惩罚系数，>1.0 降低重复，<1.0 增加重复
        text_repetition_window_size: 重复惩罚窗口大小
        debug_print_top5: 是否打印 top 5 tokens 的调试信息
        
        采样策略：
        1. 首先用原始 logits（应用 temperature）对所有 token 进行一次采样
        2. 如果采到 chunk_eos，直接返回（保持模型对"何时停止"的原始判断）
        3. 如果没采到 chunk_eos，将其屏蔽（logit 设为 -inf），继续文本 token 的采样
        4. 对文本 token 应用 repetition penalty、top-k、top-p 等策略进行最终采样
        """
        
        logits = logits.clone()
        if mode == "greedy" and temperature != 1.0:
            print("⚠️ Warning: temperature has no effect in greedy mode, ignoring it.")
        
        # ======== 0. 提前对 chunk_eos 进行独立采样判断 ========
        eos_id = self.chunk_eos_id
        
        # 根据 mode 决定是否使用随机采样
        with torch.no_grad():
            if mode == "greedy":
                # greedy 模式：只有当 chunk_eos 是 argmax 时才返回
                sampled_token = torch.argmax(logits[0]).item()
                if debug_print_top5:
                    original_probs = F.softmax(logits[0], dim=-1)
                    p_chunk_eos = original_probs[eos_id].item()
                    print(f"🎯 Greedy 判断: argmax_token={sampled_token}, P(chunk_eos)={p_chunk_eos:.6f}")
            else:
                # sampling 模式：使用随机采样
                original_probs = F.softmax(logits[0], dim=-1)
                sampled_token = torch.multinomial(original_probs, num_samples=1).item()
                if debug_print_top5:
                    p_chunk_eos = original_probs[eos_id].item()
                    print(f"🎲 提前采样判断: sampled_token={sampled_token}, P(chunk_eos)={p_chunk_eos:.6f}")
            
            # 如果采到 chunk_eos，直接返回
            if sampled_token == eos_id:
                next_token_id = torch.tensor([eos_id], device=logits.device)
                next_token_str = self.tokenizer.decode(next_token_id)
                if debug_print_top5:
                    print(f"✅ 采到 chunk_eos，直接返回: {next_token_str}")
                
                return next_token_id
        
        logits[0, eos_id] = -float('inf')
        
        if debug_print_top5:
            print("❌ 没采到 chunk_eos，继续文本采样（chunk_eos 已被屏蔽）")
            
        # 打印施加 repetition penalty 之前的 topk logits
        if debug_print_top5:
            print("🔵"*30)
            print("【BEFORE repetition penalty】施加重复惩罚之前的 Top-k logits")
            logits_before_penalty = logits[0] / temperature if mode == "sampling" else logits[0]
            topk_logits_before, topk_indices_before = torch.topk(logits_before_penalty, k=min(5, logits_before_penalty.size(-1)))
            
            for i, (token_id, logit_val) in enumerate(zip(topk_indices_before.tolist(), topk_logits_before.tolist())):
                token_str = self.tokenizer.decode([token_id])
                # 特殊处理一些token的显示
                if token_str == '\n':
                    display_str = '\\n'
                elif token_str == ' ':
                    display_str = '[SPACE]'
                elif token_str == '':
                    display_str = '[EMPTY]'
                elif token_str == '\t':
                    display_str = '\\t'
                else:
                    display_str = token_str
                
                # 标记特殊token
                special_mark = ""
                if token_id == self.listen_id:
                    special_mark = " 🎧[LISTEN]"
                elif token_id == self.tokenizer.eos_token_id:
                    special_mark = " 🛑[EOS]"
                    
                print(f"  {i+1:2d}. {display_str:10s}{special_mark:15s} (id={token_id:5d}): logit={logit_val:.4f}")
            print("🔵"*30)

        if text_repetition_penalty != 1.0 and len(self.generated_tokens) > 0:
            recent_tokens = self.generated_tokens[-text_repetition_window_size:]
            
            # make it unique
            recent_tokens = list(set(recent_tokens))
            
            # 对重复的 tokens 应用惩罚
            for token_id in recent_tokens:
                if token_id < logits.size(-1):  # 确保 token_id 在词汇表范围内
                    if text_repetition_penalty > 1.0:
                        # 惩罚重复：降低 logits
                        logits[0, token_id] /= text_repetition_penalty
                    else:
                        # 鼓励重复：增加 logits
                        logits[0, token_id] *= (1.0 / text_repetition_penalty)

        if listen_prob_scale != 1.0:    # 对 listen token 单独修改其 logit
            logits[0, self.listen_id] *= listen_prob_scale
        
        listen_rank = (logits[0] > logits[0, self.listen_id]).sum().item()
        
        # 打印 top 5 tokens（如果启用）
        if debug_print_top5:
            # 先打印 softmax 之前的 top-k logits
            logits_before_softmax = logits[0] / temperature if mode == "sampling" else logits[0]
            top5_logits_before, top5_indices_before = torch.topk(logits_before_softmax, k=min(5, logits_before_softmax.size(-1)))
            
            print("="*20)
            
            print("\n📊 Top 5 tokens BEFORE softmax (temperature={:.2f}, mode={}):".format(temperature, mode))
            for i, (token_id, logit_val) in enumerate(zip(top5_indices_before.tolist(), top5_logits_before.tolist())):
                token_str = self.tokenizer.decode([token_id])
                # 特殊处理一些token的显示
                if token_str == '\n':
                    display_str = '\\n'
                elif token_str == ' ':
                    display_str = '[SPACE]'
                elif token_str == '':
                    display_str = '[EMPTY]'
                elif token_str == '\t':
                    display_str = '\\t'
                else:
                    display_str = token_str
                
                # 标记特殊token
                special_mark = ""
                if token_id == self.listen_id:
                    special_mark = " 🎧[LISTEN]"
                elif token_id == self.tokenizer.eos_token_id:
                    special_mark = " 🛑[EOS]"
                    
                print(f"  {i+1}. {display_str:10s}{special_mark:15s} (id={token_id:5d}): logit={logit_val:.4f}")
            
            # 再打印 softmax 之后的 top-k probs
            probs = F.softmax(logits[0] / temperature if mode == "sampling" else logits[0], dim=-1)
            top5_probs, top5_indices = torch.topk(probs, k=min(5, probs.size(-1)))
            
            print("\n📊 Top 5 tokens AFTER softmax (temperature={:.2f}, mode={}):".format(temperature, mode))
            for i, (token_id, prob) in enumerate(zip(top5_indices.tolist(), top5_probs.tolist())):
                token_str = self.tokenizer.decode([token_id])
                # 特殊处理一些token的显示
                if token_str == '\n':
                    display_str = '\\n'
                elif token_str == ' ':
                    display_str = '[SPACE]'
                elif token_str == '':
                    display_str = '[EMPTY]'
                elif token_str == '\t':
                    display_str = '\\t'
                else:
                    display_str = token_str
                
                # 标记特殊token
                special_mark = ""
                if token_id == self.listen_id:
                    special_mark = " 🎧[LISTEN]"
                elif token_id == self.tokenizer.eos_token_id:
                    special_mark = " 🛑[EOS]"
                    
                print(f"  {i+1}. {display_str:10s}{special_mark:15s} (id={token_id:5d}): {prob:.4f} ({prob*100:.2f}%)")
            
            # 如果 listen token 不在 top 5，也显示它的概率
            if self.listen_id not in top5_indices.tolist():
                listen_prob = probs[self.listen_id].item()
                print(f"  ... <|listen|> 🎧 rank={listen_rank+1}, prob={listen_prob:.6f} ({listen_prob*100:.4f}%)")

        if listen_top_k is not None and listen_rank < listen_top_k:
            next_token_id = torch.tensor([self.listen_id], device=logits.device)
            next_token_str = self.tokenizer.decode(next_token_id)
            print('next_token (forced listen):', next_token_str)

            if next_token_str == "<|listen|>":
                self.context += ' '
            else:
                self.context += next_token_str

            # listen token 不添加到跟踪列表
            return next_token_id
        
        if mode == "greedy":
            next_token_id = torch.argmax(logits, dim=-1)
        elif mode == "sampling":
            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            raise ValueError("Unsupported decode mode")

        if next_token_id.item() not in self.special_token_ids:
            self.generated_tokens.append(next_token_id.item())
        else:
            self.generated_special_tokens.append(next_token_id.item())
        
        return next_token_id
