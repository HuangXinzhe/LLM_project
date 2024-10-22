# RAG概念介绍
1. 使用LLM没有训练的数据来提升LLM性能
2. 实现大致分为四个步骤
    - Embedding：使用embedding模型对文档进行embedding操作，比如OpenAI的text-Embedding-ada-002或S-BERT（https://arxiv.org/abs/1908.10084）。将文档的句子或单词块转换为数字向量。就向量之间的距离而言，彼此相似的句子应该很近，而不同的句子应该离得更远；
    - Vector Store：embedding文档之后就可以把它们存储在矢量存储中，比如ChromaDB、FAISS或Pinecone。矢量存储就像一个数据库，对矢量嵌入进行索引和存储，以实现快速检索和相似性搜索；
    - Query：文档已经嵌入并存储，向LLM提出特定问题时会embedding查询，并在向量存储中找到余弦相似度最接近问题的句子；
    - Answering Your Question：一旦找到最接近的句子，它们就会被注入到Prompt中利用LLM生成答案。
3. 单词对话不存在对话历史的问题
4. 多轮对话存在对话历史的问题（还需要考虑模型可以接受的上下文长度）