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
  
Muti Query Attention (MQA)  
k和v都是1个head，q是多个head  
encoder速度没有提升1.7-1.5ms，inference上提升明显（encoder+decoder），46-3.8ms，  
猜测原因，1.KV cache比较小，读取数据快，计算量小，2.inference阶段要顺序执行  
在MHA中，query,key,value每个向量均有768维度，而在MQA中，只有query是768维，而key和value均只剩下64维了，恰好是1个head_dim的维度。因此，我们更可以确认：在MQA中，除了query向量还保存着12个头，key和value向量都只剩1个公共头了  
  
Attention with Linear Bias（ALiBi）  
解决 transformer 训练和推理时文本长度不一致的难题  
模型在接收输入时直接去掉Position Embedding向量，而是在 Attention中计算query·Key的值后面加入一个偏置常量（非训练变量），来达到注入位置信息的效果。  
  
flash attention，细节  
通过减少访问HBM(high bandwidth memory)和on-chip SRAM内存读写时间，提高计算速度的方法。具体来说，从HBM中加载输入数据，在SRAM中执行所有的计算操作(矩阵乘法，mask，softmax，dropout，矩阵乘法)，再将计算结果写回到HBM中，分块后的部分可以在一个CUDA kernel完成，具体可以归纳为：  
1.通过分块计算，增大每次计算矩阵的最小单元，从而降低对HBM的读写次数，使得整体得到加速（HBM读写非常耗时）  
2.通过重计算，降低内存：被丢弃的变量在反传的过程中会再次使用到，需要重新计算得到，类似于梯度检查。  
  
fp16和bf16的区别  
FP32单精度浮点数，8bit 表示指数，23bit 表示小数  
BF16 是对FP32截断数据，即用8bit 表示指数，7bit 表示小数。  
FP16半精度浮点数，用5bit 表示指数，10bit 表示小数；  
与32位相比，采用BF16/FP16吞吐量可以翻倍，内存需求可以减半。但是这两者精度上差异不一样，BF16 可表示的整数范围更广泛，但是尾数精度较小；FP16 表示整数范围较小，但是尾数精度较高。  
BF16 计算时可避免计算溢出，出现Inf case；  
FP16 在输入数据超过65506 时，计算结果溢出，出现Inf case  
BF16的指数和FP32一样是为了方便混合精度计算  
  
训练的trick  
多伦对话的训练，将多伦对话进行拼接，每个回答都做掩码，这样一次训练可以达到多轮效果  
  
显卡，batch细节计算  
当是fp16时，模型2n，梯度gradient，2n，Adam状态optimizer states，12n（fp32的模型参数备份，fp32的momentum和fp32的variance）  
zero1，os  
zero2，os+gradient  
zero3，os+gradient+model  
  
lora中的alpha是什么意思  
lora训练的时候一般会增加学习率, 系数为 lora_alpha/lora_r，https://zhuanlan.zhihu.com/p/646831196

lora矩阵参数初始化  
矩阵a是降维矩阵，b是升维矩阵。 
初始化：矩阵A是 Uniform 初始化，B 是零初始化，这样最初的 lora 权重为 0，所以 lora 参数是从头学起，并没有那么容易收敛  

lora和ptuning的区别， 细节  
lora的动机是什么  
lora是大模型的低秩适配器，或者就简单的理解为适配器，模型是过参数化的，它们有更小的内在维度，模型主要依赖于这个低的内在维度（low intrinsic dimension）去做任务适配  
  
lora，大的参数矩阵，也是交transformer-block，采用2个较小的维度的矩阵相乘做支路，再相加，微调时，只调试支路参数  
ptuning，主要针对NLU任务，对于BERT类双向语言模型采用模版(P1, x, P2, [MASK], P3)，对于单向语言模型采用(P1, x, P2, [MASK])，ptuningv2，在每层都加上prompt参数  
Prefix-tuning是做生成任务，它根据不同的模型结构定义了不同的Prompt拼接方式，在GPT类的自回归模型上采用[PREFIX, x, y]，在T5类的encoder-decoder模型上采用[PREFIX, x, PREFIX', y]：  

并行技术  
数据并行、张量并行和流水线并行  
1.数据并行 Data Parallelism，作为最简单且常用的方式，数据并行将相同的模型权重复制到多个设备，并将一部分数据分配给每个设备同时处理，相当于沿Batch维度对训练过程进行并行化
DP (Data Parallel)  
本质上是单进程多线程的实现方式，只能实现单机训练不能算是严格意义上的分布式训练。步骤如下：  
首先将模型加载到主GPU上，再复制到各个指定从GPU；  
将输入数据按照Batch维度进行拆分，各个GPU独立进行forward计算；  
将结果同步给主GPU完成梯度计算和参数更新，将更新后的参数复制到各个GPU。  
主要存在的问题：  
负载不均衡，主GPU负载大  
采用 PS 架构通信开销大  
DDP (Distribution Data Parallel)   
采用 AllReduce 架构，在单机和多机上都可以使用。负载分散在每个gpu节点上，通信成本是恒定的，与 GPU 数量无关。  
2.张量并行 Tensor Parallelism   
张量并行指的是将一个张量（tensor）沿特定维度分成若干部分在不同的设备上分别计算 ，下面以Transformer结构为例介绍这种并行方式   
3.流水线并行 Pipeline Parallelism  
流水线并行是一种通过将模型并行与数据流水线相结合来加速神经网络训练的方法。其核心思想是，模型按层分割成若干块，每块都交给一个设备。在前向传递过程中，每个设备将中间的激活传递给下一个阶段。在后向传递过程中，每个设备将输入张量的梯度传回给前一个流水线阶段  

遇到困难，怎样克服，亮点有哪些：  
搜索ner，开始用的bert，发现效率更不上，我们需要性能更强的模型，而且需要控制成本，线上不会大量用gpu，所以我们用了一个cnn的小模型，参数只有bert的10分之1，单次请求时间是6ms左右  
但是由于泛化性差一些，做了一些改进  
1.加大训练数据  
2.加入词特征信息  
3.自己训练word2vec的embedding  
效果有所提升  
大模型  
1.开源代码，全参数微调，微调后输出混乱，发现预处理代码有问题，mask后的编码specialtoken位置不对导致，修改后好了  
2.全参数微调，一开始我们全部用的领域内数据，发现遗忘问题严重，后来参考论文，加入通用数据，1:5，wiki百科，qa对话等  
