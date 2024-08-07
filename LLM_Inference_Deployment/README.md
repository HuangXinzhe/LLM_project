# 大模型推理部署
参考资料：https://mp.weixin.qq.com/s/xIbNSAI9cKGIA19yZhIEgg  

- vLLM：适用于大批量Prompt输入，并对推理速度要求高的场景；
- Text generation inference：依赖HuggingFace模型，并且不需要为核心模型增加多个adapter的场景；
- CTranslate2：可在CPU上进行推理；
- OpenLLM：为核心模型添加adapter并使用HuggingFace Agents，尤其是不完全依赖PyTorch；
- Ray Serve：稳定的Pipeline和灵活的部署，它最适合更成熟的项目；
- MLC LLM：可在客户端（边缘计算）（例如，在Android或iPhone平台上）本地部署LLM；
- DeepSpeed-MII：使用DeepSpeed库来部署LLM
- TensorRT-LLM：英伟达(NVIDIA)在TensorRT基础上针对LLM优化所推出的推理加速引擎TensorRT-LLM
- GPT4All