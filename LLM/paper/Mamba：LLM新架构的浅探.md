# Mamba：LLM新架构的浅探
资料地址：https://mp.weixin.qq.com/s/JnMYTax6UbQHCAzg-7_P-w
Mamba模型（https://github.com/state-spaces/mamba）：
## Mamba简介
Mamba是LLM的一种新架构，与Transformers等传统模型相比，它能够更有效地处理长序列。它利用选择性状态空间模型（SSM），根据内容动态过滤和处理信息，允许模型选择性地记住或忽略输入的部分。Mamba在处理速度和缩放能力方面有了显著改进，尤其是在较长序列的情况下。
```
# 加载模型
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments

# Load model
model = MambaLMHeadModel.from_pretrained(  
    "state-spaces/mamba-1.4b",   
    device="cuda",   
    dtype=torch.bfloat16
    )

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")



# 使用简单Prompt完成续写任务
prompt=\
"""A conversation between a user and a smart AI assistant.

### User: Hello!
### Assistant:
"""

prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")

# from https://github.com/state-spaces/mamba/blob/main/benchmarks/benchmark_generation_mamba_simple.py#L54
output_tokenized = model.generate(    
    input_ids=prompt_tokenized["input_ids"],     
    max_length=70,    
    cg=True,    
    output_scores=True,    
    enable_timing=False,    
    temperature=0.7,    
    top_k=40,    
    top_p=0.1,    
    )
output=tokenizer.decode(output_tokenized[0])

print(output)
```
## 微调Mamba
数据集：使用高质量的ChatML多轮对话数据集Open Assistant数据集（https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25）

微调过程：
- Tokenizing数据集
- 定义collate函数
- 使Mamba适应Hugging Face Trainer，由于Mamba独特的架构，需要修改一些代码。
```
# 加载数据集并对其tokenize
from datasets import load_dataset

dataset=load_dataset("OpenAssistant/oasst_top1_2023-08-25")
"""
数据集有13k条样本，并且已经划分好了训练集和测试集
数据集中的大多数对话（92%）有少于1000个tokens组成。因此tokenize过程中，将每个会话截断为1024个tokens就足够了
"""



import os 

def tokenize(element):    
    return tokenizer(        
        element["text"],        
        truncation=True,        
        max_length=1024,        
        add_special_tokens=False,    
        )

dataset_tokenized = dataset.map(    
    tokenize,     
    batched=True,     
    num_proc=os.cpu_count(),    # multithreaded    
    remove_columns=["text"]     # don't need this anymore, we have tokens from here on
    )



# 定义collate函数
# 将数据集传入Trainer之前，由于并非所有对话的长度都相同，必须将它们分批分组，需要定义pad_token
tokenizer.pad_token = tokenizer.eos_token

# collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }def collate(elements):    
    tokenlist=[e["input_ids"] for e in elements]    tokens_maxlen=max([len(t) for t in tokenlist])
    
    input_ids,labels = [],[]    
    for tokens in tokenlist:   
        pad_len=tokens_maxlen-len(tokens)
        
        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content otherwise 0        
        input_ids.append(tokens + [tokenizer.pad_token_id]*pad_len)           
        labels.append(tokens + [-100]*pad_len)    

    batch={        
        "input_ids": torch.tensor(input_ids),        "labels": torch.tensor(labels),    
        }    
    return batch
# 由于Mamba没有使用注意力机制，因此批次中不包含注意力掩码



# 准备Mamba Trainer
"""
目前，Mamba还没有被添加到Hugging Face生态系统中。标准的Hugging Face Trainer需要一个包括labels的向前函数，而Mamba没有。
        为了解决这个问题，我们需要实现一个临时解决方案，通过使用monkey补丁向模型添加一个新的前向函数。这不是最优雅的方法，但在Mamba成为Hugging Face transformer库的一部分之前，这是一个临时的解决方案。
"""
# monkey patch MambaLMHeadModel.forward 
def forward_with_loss(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, labels = None):    
    """    
    "position_ids" is just to be compatible with Transformer generation. We don't use it.    num_last_tokens: if > 0, only return the logits for the last n tokens    
    """    
    hidden_states = self.backbone(input_ids, inference_params=inference_params)    
    if num_last_tokens > 0:        
        hidden_states = hidden_states[:, -num_last_tokens:]    
        lm_logits = self.lm_head(hidden_states)        
        
        # Source: https://github.com/huggingface/transformers/blob/80377eb018c077dba434bc8e7912bcaed3a64d09/src/transformers/models/llama/modeling_llama.py#L1196    
        from torch.nn import CrossEntropyLoss    
        if labels is not None:        
            logits = lm_logits        
            # Shift so that tokens < n predict n        shift_logits = logits[..., :-1, :].contiguous()        
            shift_labels = labels[..., 1:].contiguous()        
            # Flatten the tokens        
            loss_fct = CrossEntropyLoss()        
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)        
            shift_logits = shift_logits.view(-1, self.backbone.embedding.weight.size()[0])        shift_labels = shift_labels.view(-1)        # Enable model parallelism        shift_labels = shift_labels.to(shift_logits.device)        
            loss = loss_fct(shift_logits, shift_labels)        
            return (loss,)       
        else:        
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])        
            return CausalLMOutput(logits=lm_logits)
MambaLMHeadModel.forward=forward_with_loss

# patch MambaLMHeadModel
MambaLMHeadModel.forward=forward_with_loss

# (re)load model 
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-1.4b", device="cuda", dtype=torch.bfloat16)
"""
或者可以使用优秀的训练器axolotl（https://github.com/OpenAccess-AI-Collective/axolotl）或使用mamba-chat（https://github.com/havenhq/mamba-chat）进行训练。
"""



# 训练Mamba模型
from transformers import Trainer, TrainingArguments
bs=4        # batch size
ga_steps=1  # gradient acc. steps
epochs=3
steps_per_epoch=len(dataset_tokenized["train"])//(bs*ga_steps)
lr=0.

args = TrainingArguments(    
    output_dir="out",    
    per_device_train_batch_size=bs,    
    per_device_eval_batch_size=bs,    
    evaluation_strategy="steps",    
    logging_steps=1,    
    eval_steps=steps_per_epoch,    
    save_steps=steps_per_epoch,    
    gradient_accumulation_steps=ga_steps,    
    num_train_epochs=epochs,    
    lr_scheduler_type="constant",    
    learning_rate=lr,    
    group_by_length=True,    
    bf16=True,                  # mixed precision training    save_safetensors=False,     # saving will fail without this
    )

trainer = Trainer(    
    model=model,    
    tokenizer=tokenizer,    
    args=args,    
    data_collator=collate,    
    train_dataset=dataset_tokenized["train"],    
    eval_dataset=dataset_tokenized["test"],
    )

trainer.train()
```
## 评价Mamba模型
聊天机器人的评估很难，因为结果很难衡量。
什么是好的会话/指令跟随模式？这个问题的解决方案不止一种。在有人想出如何正确应用这样的东西之前，我们将不得不依赖基准（https://github.com/EleutherAI/lm-evaluation-harness）测试、聊天机器人竞技场（https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard）和人工智能裁判（https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard）。
