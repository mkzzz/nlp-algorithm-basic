## ChatGPT入门

### 1 ChatGPT背景介绍

#### 1.1 什么是ChatGPT?

ChatGPT 是由人工智能研究实验室 OpenAI 在2022年11月30日发布的全新聊天机器人模型，一款人工智能技术驱动的自然语言处理工具。它能够通过学习和理解人类的语言来进行对话，还能根据聊天的上下文进行互动，真正像人类一样来聊天交流，甚至能完成撰写邮件、视频脚本、文案、翻译、代码等任务。

#### 1.2 为什么ChatGPT这么火？

- 操作简单，容易上手，打开 ChatGPT 聊天框之后只需要在对话框里输入问题，就可以获得答案。能够应对日常对话。


- 根据官方介绍，ChatGPT 以对话方式进行交互。对话格式使 ChatGPT 能够回答后续问题、承认错误、质疑不正确的前提和拒绝不适当的请求。


简单使用ChatGPT的示例演示：

- 单轮对话完成问题解答：

  - 假如 “我想对暗恋的女孩写一篇表白信”，如何利用 ChatGPT 实现呢？

  <div align=center><img src="img/01.png" style="zoom:60%" ><img/></div>

- 多轮对话完成问题解答：

  - 假如 “当你一个人孤单的时候，想找个人聊聊天，那么ChatGPT就可以充当你的聊天对象”。
  - 第一步：你可以输入文本如：“我现在很寂寞。你能陪我聊聊天吗?”

  <div align=center><img src="img/1675317982445.png" style="zoom:60%" ><img/></div>

  - 第二步：继续输入文本如：“我的女朋友和我分手了，我心情很难过”

  <div align=center><img src="img/1675318196155.png" style="zoom:60%" ><img/></div>

  - 第三步：继续输入文本如：“我现在还忘不掉她，我是不是应该给她打个电话，重新挽回这段感情”

  <div align=center><img src="img/1675318340452.png" style="zoom:60%" ><img/></div>

  - 第四步：继续输入：“那我想问下你有女朋友吗？”

  <div align=center><img src="img/1675318641919.png" style="zoom:60%" ><img/></div>

### 2 如何使用ChatGPT

使用ChatGPT需要四个步骤：

- 第一步：注册ChatGPT账号
- 第二步：通过短信平台 sms-activate.org 完成ChatGPT手机号验证
- 第三步：登录ChatGPT账号，对话框输入文本，开始使用ChatGPT

#### 2.1如何注册ChatGPT账号

- 第一步：打开https://chat.openai.com/auth/login链接，点击注册按钮开始注册

<div align=center><img src="img/1675320384688.png" style="zoom:65%" ><img/></div>

- 第二步：通过邮箱注册并完成邮箱验证：输入邮箱用以接收验证链接，可以输入 qq或者是163 等邮箱。然后点击`Continue`，输入密码，接着点`Continue`，然后会发一封邮件给邮箱，到邮箱中点击确认即可

<div align=center><img src="img/1675320766171.png" style="zoom:65%" ><img/></div>

<div align=center><img src="img/1675321497885.png" style="zoom:65%" ><img/></div>

- 第三步：邮箱验证完之后，接下来要填入手机号验证。
  - **注意**：这里不能是使用国内的手机号，因为 OpenAI 服务目前暂不支持国内，因此国内手机号是没有办法通过验证的，但是我们可以借助短信平台完成手机号验证。可参考步骤 2.2 实现手机号验证。

<div align=center><img src="img/1675323722859.png" style="zoom:64%" ><img/></div>

#### 2.2 完成手机号验证

接下来，我们将借助sms-activate.org平台实现手机号验证，这个平台需要我们用户付费使用，本次通过该平台实现手机号验证大约需要花费2元左右。以下是具体实现步骤：

- 第一步，访问短信平台网址 https://sms-activate.org/cn/，完成注册和登录

<div align=center><img src="img/1675324170411.png" style="zoom:56%" ><img/></div>

- 第二步：点击余额进行充值

<div align=center><img src="img/1675324469058.png" style="zoom:56%" ><img/></div>

- 第三步：选择支付宝进行充值

<div align=center><img src="img/1675324594500.png" style="zoom:56%" ><img/></div>

<div align=center><img src="img/1675324870937.png" style="zoom:65%" ><img/></div>

- 第四步：充值完后，在左侧栏目栏中输入 openai  进行搜索

<div align=center><img src="img/1675325284576.png" style="zoom:55%" ><img/></div>

- 第五步：选择一个国家的服务，进行手机号码验证，这里选用的是印度地区（你也可以选用其他地区，但要注意价格区别）

<div align=center><img src="img/1675325704626.png" style="zoom:55%" ><img/></div>

- 第六步：买入支付后，下图红色方框内就是号码了，把他输入进注册页，然后点击Send code 接受验证码

<div align=center><img src="img/1675325973601.png" style="zoom:70%" ><img/></div>

<div align=center><img src="img/1675326206080.png" style="zoom:65%" ><img/></div>

- 第七步：等几分钟后，就可以顺利接收到验证码了（注意：如果很久接不到验证码可以手动叉掉换一个国家的号码再试试。），然后将验证码输入确认框Enter code完成注册

<div align=center><img src="img/1675326405842.png" style="zoom:65%" ><img/></div>

<div align=center><img src="img/1675326504277.png" style="zoom:65%" ><img/></div>

- 第八步：如果出现以下界面，代表ChatGPT账号注册成功

<div align=center><img src="img/1675326648551.png" style="zoom:65%" ><img/></div>

**注意**：如果在上述注册过程中，出现如下界面，需要切换你的网络代理地区，比如切换为：美国、韩国等可自行尝试

<div align=center><img src="img/1675326778413.png" style="zoom:65%" ><img/></div>

#### 2.3 使用 ChatGPT 

- 用上述注册的账号，打开https://chat.openai.com/auth/login网址进行登录

<div align=center><img src="img/1675327078930.png" style="zoom:65%" ><img/></div>

<div align=center><img src="img/1675327325448.png" style="zoom:65%" ><img/></div>

- 登录后，即可进入ChatGPT聊天框，实现文本对话

<div align=center><img src="img/1675327503852.png" style="zoom:55%" ><img/></div>



### 3 ChatGPT入门程序（python直接调用ChatGPT）

假如你是一名程序员，在工作中想使用ChatGPT来完成一些任务，但是使用网页交互模式难免有些麻烦，那么此时我们可以使用代码的方式，实现ChatGPT模型的调用，其功能和网页输入框效果基本一致。

以下是使用python调用ChatGPT模型的步骤，一共需要三步：

- 第一步：你需要先访问OpenAI的API网站(https://openai.com/api/)，获取你的 API 密钥
- 第二步：安装openai第三方库
- 第三步：使用python调用ChatGPT模型API

#### 3.1 API密钥获取实现：

- 第一步：利用你注册的ChatGPT账号，登录OpenAI的API网站(https://openai.com/api/)

<div align=center><img src="img/1675328180132.png" style="zoom:50%" ><img/></div>

- 第二步：登录进入后，点击Personal，选择View API keys

<div align=center><img src="img/1675328386036.png" style="zoom:50%" ><img/></div>

- 第三步：创建你的API key 并复制使用

<div align=center><img src="img/1675328617729.png" style="zoom:55%" ><img/></div>

<div align=center><img src="img/1675328689162.png" style="zoom:65%" ><img/></div>

#### 3.2 安装OpenAI的Python库，可以使用以下命令完成：

```python
pip install openai
```

#### 3.3 python实现ChatGPT模型代码实现：

```python
import openai

openai.api_key = "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"

model_engine = "text-davinci-002"

prompt = "请帮我利用python实现下二分查找算法的代码" # prompt代表用户输入文本

completions = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.7,
)

message = completions.choices[0].text
print(message)


#########################################################
以下为 print 输出结果展示：
# 功能：二分查找
# 输入参数：array：数组，number：查找数字
# 输出参数：查询数字在数组中的下标，如果查询不到返回-1
def binary_search(array, number):
	# 定义左右指针
	left = 0
	right = len(array) - 1 

	# 二分查找
	while left <= right:
		# 求中间下标
		mid = (left + right) // 2

		# 如果查询数字小于数组中间的数字，则查询范围变为左边的数组
		# 否则查询范围变为右边的数组
		if number < array[mid]:
			right = mid - 1
		elif number > array[mid]:
			left = mid + 1
		else:
			return mid

	return -1
```

**请注意**，上述代码仅是示例，您可能需要根据自己的需要调整代码以获得所需的结果

### 4 ChatGPT应用场景

ChatGPT模型的应用场景主要分为以下三大模块：

- 代码相关：如程序语言转换，比如Python转Java、生成程序命令、修复代码Bug、程序代码解释等;
- 日常生活相关：如语言翻译；美食制作配方生成；段落关键字提取；餐馆点评；
- 职业相关：创作一个短故事；产品广告设计；生成面试问题帮助面试者准备面试

下面我们以日常生活相关的场景，来实现ChatGPT功能的应用。这里我们分别选取一道中国菜：鱼香肉丝，一道美国菜：hamburger，应用ChatGPT实现美食制作。

- 中国菜“鱼香肉丝”ChatGPT给出的做法

<div align=center><img src="img/1675330076085.png" style="zoom:60%" ><img/></div>

- 美国菜“hamburger”ChatGPT给出的做法

<div align=center><img src="img/1675330310673.png" style="zoom:60%" ><img/></div>

### 5 如何使用chatgpt实现真正的AI项目？大家关注后续chatgpt完成教程

