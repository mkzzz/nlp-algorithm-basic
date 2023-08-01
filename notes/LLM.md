LLM常用参数  
temperature/top-k/top-p都是采样参数，如果 k 和 p 都启用，则 p 在 k 之后起作用  
temperature，于控制模型输出的结果的随机性，这个值越大随机性越大，等于0时每次输出结果一样，此参数会使分布更加平滑，值越小越平滑。用在softmax上时，e**x转换成e**(x/t)  
topk，筛选候选词，topk，就是按照概率高低选前k个  
topp，也叫核采样（Nucleus Sampling），候选词概率加起来大于等于topp的值后，把后面的候选去掉  
  
chatglm  
可以分为三种方式：单词（MASK）、句子（sMASK）、文档（gMASK）  
special_tokens:{'[MASK]': 64789, '[gMASK]': 64790, '[sMASK]': 64791, 'sop': 64792, 'eop': 64793}  
<div align="center"><img src="../assets/glm1.jpeg"></div>  

chatglm6bv2
对比chatglmv1提升如下：  
1）1.4T 中英标识符的预训练与人类偏好对齐训练  
2）基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K  
3）基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K  

llama
7B、13B、33B、65B 四种版本，与原始的 transformer Decoder 相比，LLaMA主要有以下改进：  
1）LLaMA对每个transformer子层的输入进行归一化，而不是对输出进行归一化。同时使用RMSNorm归一化函数  
2）LLaMA用SwiGLU激活函数取代ReLU非线性，以提高性能  
3）LLaMA删除了绝对位置嵌入，取而代之的是在网络的每一层添加旋转位置嵌入（RoPE）  

llama2  
7B、13B 和 70B三种版本，7B & 13B 使用与 LLaMA 1 相同的架构，并且是商业用途的 1 对 1 替代  
相比llama1有如下改进：  
1）llama2 模型接受了 2 万亿个标记的训练，训练语料相比LLaMA多出40%  
2）llama2上下文长度是llama1的两倍，上下文长度是由之前的2048升级到4096，可以理解和生成更长的文本  
3）从人类反馈中强化学习，除了Llama 2版本，还发布了LLaMA-2-chat，接受了超过 100 万个新的人类注释的训练，使用来自人类反馈的强化学习来确保安全性和帮助性。
  
参数显存占用计算  
chatgln6B，fp16，6b*2=12g  
llama7b，fp32，7b*4=28g
