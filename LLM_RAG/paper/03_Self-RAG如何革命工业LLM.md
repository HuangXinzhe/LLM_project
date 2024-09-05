# Self-RAG如何革命工程师的LLM
资料地址：https://mp.weixin.qq.com/s/a3tzatoHdHde9-IqCIwWHQ
论文地址：https://arxiv.org/pdf/2310.11511.pdf
Github地址：https://github.com/AkariAsai/self-rag

## RAG技术中存在的问题
1. 检索到的前k个文本不包含问题的所有答案
2. 检索到的相似度较高的文本不总是产生相关上下文
3. 超过一定范围的问题回答不了

## Self-RAG
通过按需检索和自我发丝来改进LLM的生成质量
- 训练一个LM，使能够反思自己的生成过程，并生成任务输出和中间的特殊tokensreflection tokens）（比如[Retrieval], [No Retrieval], [Relevant], [Irrelevant], [No support / Contradictory], [Partially supported], [Utility]等）。这些reflection(反思) tokens被分类为检索tokens和批评tokens，分别表示需要检索的需求和其生成质量。

 "SELF-RAG" 的系统中使用的四种反思tokens的类型：
 ① Retrieve：这是一个决策过程，它决定了是否从某个资源 R 中检索信息。② IsREL：这是一个相关性检查，目的是确定给定的数据 d 是否包含解决问题 x 所需的相关信息。
 ③ IsSUP：这是一个验证过程，用于检查提供的响应 y 中的声明是否得到了数据 d 的支持。
 ④ IsUSE：这是一个评估过程，旨在评估给定的响应 y 对于问题 x 有多么有用。输出是一个从1到5的评分，5分代表最有用。

 ### self-rag实施步骤
 1. 按需逐个检索文本
 2. 并行生成每个检索到的文本和提示词的结果
 3. 对输出进行评价，选择最佳的段落

 ### self-RAG推理实践
 ```
 from vllm import LLM, SamplingParams
 model = LLM("selfrag/selfrag_llama2_7b", download_dir="/gscratch/h2lab/akari/model_cache", dtype="half")
 sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
 def format_prompt(input, paragraph=None):
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
    if paragraph is not None:
    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return prompt
query_1 = "Leave odd one out: twitter, instagram, whatsapp."
query_2 = "Can you tell me the difference between llamas and alpacas?"
queries = [query_1, query_2]
# for a query that doesn't require retrieval
preds = model.generate([format_prompt(query) for query in queries], sampling_params)
for pred in preds:
    print("Model prediction: {0}".format(pred.outputs[0].text))



paragraph="""Llamas range from 200 to 350 lbs., while alpacas weigh in at 100 to 175 lbs."""
def format_prompt_p(input, paragraph=paragraph):  
    prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)  
    if paragraph is not None:    
        prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)  
    return prompt
query_1 = "Leave odd one out: twitter, instagram, whatsapp."
query_2 = "Can you tell me the differences between llamas and alpacas?"
queries = [query_1, query_2]

# for a query that doesn't require retrieval
preds = model.generate([format_prompt_p(query) for query in queries], sampling_params)
for pred in preds:  
    print("Model prediction: {0}".format(pred.outputs[0].text))



# 输出
[Irrelevant]Whatsapp is the odd one out.
[No Retrieval]Twitter and Instagram are both social media platforms, 
while Whatsapp is a messaging app.[Utility:5]

[Relevant]Llamas are larger than alpacas, with males weighing up to 350 pounds.
[Partially supported][Utility:5]
 ```

 ### 完整的self-rag实现
 https://zhuanlan.zhihu.com/p/682482995
 https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/