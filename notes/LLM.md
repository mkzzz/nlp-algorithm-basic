LLM常用参数  
temperature/top-k/top-p都是采样参数，如果 k 和 p 都启用，则 p 在 k 之后起作用  
temperature，于控制模型输出的结果的随机性，这个值越大随机性越大，等于0时每次输出结果一样，此参数会使分布更加平滑，值越大越平滑。用在softmax上时，e**x转换成e**(x/t)  
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

langchain  
LangChain是一个强大的框架，旨在帮助开发人员使用语言模型构建端到端的应用程序。它提供了一套工具、组件和接口，可简化创建由大型语言模型 (LLM) 和聊天模型提供支持的应用程序的过程。LangChain 可以轻松管理与语言模型的交互，将多个组件链接在一起，并集成额外的资源，例如 API 和数据库  
LangChain 旨在为六个主要领域的开发人员提供支持：  
1）LLM 和提示：LangChain 使管理提示、优化它们以及为所有 LLM 创建通用界面变得容易。此外，它包括一些用于处理 LLM 的便捷实用程序  
2）链(Chain)：这些是对 LLM 或其他实用程序的调用序列。LangChain 为链提供标准接口，与各种工具集成，为流行应用提供端到端的链。  
3）数据增强生成：LangChain 使链能够与外部数据源交互以收集生成步骤的数据。例如，它可以帮助总结长文本或使用特定数据源回答问题。  
4）Agents：Agents 让 LLM 做出有关行动的决定，采取这些行动，检查结果，并继续前进直到工作完成。LangChain 提供了代理的标准接口，多种代理可供选择，以及端到端的代理示例。  
5）内存：LangChain 有一个标准的内存接口，有助于维护链或代理调用之间的状态。它提供了一系列内存实现和使用内存的链或代理的示例  
6）评估：很难用传统指标评估生成模型。这就是为什么 LangChain 提供提示和链来帮助开发者自己使用 LLM 评估他们的模型。  
  
autogpt  
可以理解为作者设计了一个十分精巧的prompt，然后把我们要执行的命令，基于prompt模板封装后发给GPT-4，然后根据结果来执行。  
其核心在于它把我们的命令发送给GPT-4的时候，让GPT-4根据指定的COMMAND来选择操作，上述COMMAND中，大家可以看到包括谷歌搜索、浏览网站、读写文件、执行代码等。AutoGPT会把问题，如“寻找今天推特上最火的AI推文”发给GPT-4，并要求GPT-4根据这些COMMAND选择最合适的方式去得到答案，并给出每一个COMMAND背后需要使用的参数，包括URL、执行的代码等  

sequential modulelist区别  
nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的  
nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器，直接用python的list时，不会自动注册到网络中，可以使用 forward 来计算输出结果。但是如果用其实例化的网络进行训练的时候，因为这些层的parameters不在整个网络之中，所以其网络参数也不会被更新，也就是无法训练  
不同点:Sequential有先后顺序，且内部实现了forward函数，ModuleList没有先后顺序，没有实现内部forward函数  
  
混合精度为什么在优化器中是32位的  
半精度累加误差会积累  
  
torch2.0有什么改进  
即torch.compile。这是一个新增的特性，可以极大地提升模型运行速度  
