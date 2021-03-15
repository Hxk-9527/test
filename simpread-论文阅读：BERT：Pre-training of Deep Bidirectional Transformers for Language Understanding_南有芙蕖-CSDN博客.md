> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_42691585/article/details/108956159)

论文阅读：BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding 预训练的深度双向 Transformer 语言模型
==============================================================================================================

>   原文链接：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805?context=cs)  
>   作者：**Jacob Devlin,Ming-Wei Chang,Kenton Lee,Kristina Toutanova**

>   总结的论文思维导图如下：  
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201010124637548.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)

### 目录

*   [论文阅读：BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding 预训练的深度双向 Transformer 语言模型](#BERTPretraining_of_Deep_Bidirectional_Transformers_for_Language_Understanding___Transformer__0)
*   *   [摘要](#_8)
    *   [1 简介](#1__11)
    *   [2 相关工作](#2__22)
    *   *   [2.1 无监督的基于特征的方法](#21__24)
        *   [2.2 无监督的微调方法](#22__30)
        *   [2.3 监督数据的迁移学习](#23__34)
    *   [3 BERT](#3_BERT_36)
    *   *   [3.1 预训练BERT](#31_BERT_57)
        *   [3.2 微调BERT](#32_BERT_70)
    *   [4 实验](#4__76)
    *   *   [4.1 GLUE](#41_GLUE_78)
        *   [4.2 SQuAD v1.1](#42_SQuAD_v11_90)
        *   [4.3 SQuAD v2.0](#43_SQuAD_v20_104)
        *   [4.4 SWAG](#44_SWAG_119)
    *   [5 消融研究](#5__128)
    *   *   [5.1 预训练任务的效果](#51__130)
        *   [5.2 模型大小的影响](#52__144)
        *   [5.3 基于特征的BERT方法](#53_BERT_152)
    *   [6 结论](#6__160)
    *   [附录](#_163)

摘要
--

>   我们引入了一种新的语言表示模型叫为BERT，它用Transformers的双向编码器表示（Bidirectional Encoder Representations）。与最近的语言表示模型（Peters et al.，2018; Radford et al.，2018）不同，**BERT通过在所有层的上下文联合调节来预训练深层双向表示（deep bidirectional representations）**。因此，预训练的BERT可以添加一个**额外的输出层进行微调**，可以在广泛的任务上产生目前最好的效果，例如问答和语言推理，而不需要对大量特定任务的结构进行修改。  
>   BERT在概念上简单且经验丰富。它在11项自然语言处理任务中获得了最新的成果，包括将**GLUE得分提高到80.5％（绝对值提高7.7％），MultiNLI准确度提高到86.7％（绝对值提高4.6％），SQuAD v1.1问答测试F1达到93.2（绝对值提高1.5分）和SQuAD v2.0测试F1达到83.1（绝对值提高5.1点）**。

1 简介
----

  语言模型预训练已经证明对改善许多自然语言处理任务是有效的。这些任务包括句子级任务，如自然语言推理和释义，旨在**通过对句子进行整体分析来预测句子之间的关系**，以及词级任务，如命名实体识别和问答，其中**模型需要在词级别有细粒度的输出**。

  在预训练的语言表示应用到下游任务上有两种策略:**基于特征和微调**。**基于特征的方法**，如ELMo使用特定于任务的体系结构，将预训练表示作为额外特征。**微调方法**，如生成式预训练transformer（OpenAI GPT），引入了最小的特定于任务的参数，并通过简单地微调所有预训练参数来训练下游任务。在以前的工作中，**这两种方法在训练前都有相同的目标函数，它们使用单向语言模型来学习通用语言表达**。

  我们认为，当前的技术严格限制了预训练表示的能力，特别是微调方法。**主要的限制是标准语言模型是单向的，在预训练期间限制了架构的选择**。例如，在OpenAI GPT中，作者使用了一种从左到右的架构，其中每个词只能关注transformer self-attention层中的先前词。这种限制对于句子级别的任务来说不是最佳的，当对SQuAD之类的词级别任务应用基于微调的方法时，这种限制可能会带来毁灭性的影响，其中从两个方向应用语言模型非常重要。

  在本文中，我们通过transformer提出BERT :双向编码器表示来改进基于微调的方法。BERT受到**完形填空**的启发，通过提出一个**新的预训练对象“masked language model”( MLM )，来解决上面提到的单向约束**。Masked language model在输入中随机屏蔽掉一些词，**目的是仅根据屏蔽单词的上下文预测其原始词汇id**。与从左到右的语言模型预训练不同，**MLM目标允许表示融合左右上下文，这允许我们预训练一个深度双向transformer**。除了Masked language model之外，我们还引入了一个“next sentence prediction”任务，共同预训练文本对表示。本文的贡献如下:

> *   我们展示了语言表达的双向预训练的重要性。**BERT使用Masked language model来实现预训练的深度双向表示**。这也与Peters（2018）等人形成对比，它使用独立训练的从左到右和从右到左LMs的简单串联。
> *   我们表明，预训练的表示消除了许多精心设计的任务特定体系结构的需求。**BERT第一个基于微调的表示模型**，在大量的句子级和符号级任务上实现了最先进的性能，优于许多特定于任务的架构。
> *   BERT提高了11项NLP任务的最新水平。我们还展示了BERT的广泛应用，证明了我们模型的双向性质是最重要的新贡献。代码和预训练的模型可在[https://github.com/google-research/bert](https://github.com/google-research/bert)上获得。

2 相关工作
------

  **预训练通用语言表达**已经有很长的历史，我们简要回顾了本节中最流行的方法。

### 2.1 无监督的基于特征的方法

  学习可以广泛应用的词表示已经热门了数十年，包括神经网络方法和无神经方法。**预训练词嵌入**是现代NLP系统必不可少的部分，与从头开始学习的嵌入相比提供了显着的性能提高。为了预训练词嵌入向量，已经使用**上下文的语言建模**目标，以及**在上下文中区分正确和错误单词**的目标。

  这些方法已经被一般化为更大粒度的表示，如**句子嵌入或段落嵌入**。为了训练句子表示，先前的工作已经使用目标来对下一个句子进行排序，**从左到右生成下一个句子的单词，给出了前一句子的表示，或对自动编码器派生的目标进行降噪**。

  Elmo从另外一个角度概况了传统词嵌入的研究。它们**从左到右和从右到左的语言模型中提取上下文相关的特征**。每个符号的上下文表示是从左到右和从右到左表示的连接。当将上下文词嵌入与现有任务特定体系结构集成时，ELMo提出了几个主要NLP基准的最新技术，包括**问答、情感分析和命名实体识别**。Melamud等人（2016） 提出通过任务学习上下文表示，以使用LSTM从上下文预测单个词。与ELMo类似，他们的模型基于特征，而非深度双向的。Fedus等人（2018） 显示了完形填空任务可用于提高文本生成模型的鲁棒性。

### 2.2 无监督的微调方法

  与基于特征的方法一样，**第一种方法仅在未标记文本的预训练词嵌入参数上才起作用**。

  最近，产生上下文标记表示的句子或文档编码器已经从未标记的文本中进行了预训练，并针对监督下游任务进行了微调。这些方法的优势是**调整的参数很少**。由于这一优势，OpenAI GPT在**GLUE基准测试**中的许多句子级任务上取得先前最先进的结果。从左到右的语言模型和自动编码器已用于预训练此类模型。

### 2.3 监督数据的迁移学习

  还有一些工作显示了具有大型数据集的监督任务的有效迁移，例如自然语言推理和机器翻译。**计算机视觉**研究也证明了从大型预训练模型中迁移学习的重要性，其中一个有效的方法是**用ImageNet对预训练模型进行微调**。

3 BERT
------

  本节详细介绍**BERT的细节实现**。我们的框架有两个步骤：预训练和微调。在**预训练**期间，模型在不同的预训练任务上训练未标记的数据。对于**微调**，首先用预训练的参数初始化BERT模型，并使用来自下游任务的标签数据对所有参数进行微调。每个下游任务都有单独的微调模型，即使它们使用了相同的预训练参数进行了初始化。图1中的问答系统示例将作为本节的运行示例。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007214928255.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)

  BERT的一个显着特点是**它在不同任务中有着统一架构**。预训练架构和最终下游架构之间存在微小差异。

  **模型架构**

>   **BERT模型架构是基于Vaswani 等人描述的原始实现的一个多层双向的transformer的encoder，并在发布在tensor2tensor库中**。由于transformer的使用已经非常普遍并且我们的实现几乎与原始实现完全相同，我们将省略对模型架构详尽的背景描述，并向读者推荐Vaswani 等人和一些优秀的指南“The Annotated Transformer”。  
>   在这个工作中，我们**用L表示层的数量，用H表示隐藏层大小，用A表示self-attention的head数**量。我们主要报告两个模型的大小：  
>   BERTBASE（L=12, H=768, A=12, Total Parameters=110M），  
>   BERTLARGE（L=24, H=1024, A=16, Total Parameters=340M）。  
>   为了进行比较，选择的BERTBASE与OpenAI GPT具有相同的模型大小。然而，重要的是，BERT Transformer使用双向的self-attention，而GPT Transformer使用有约束的self-attention，其中每个token只能处理其左侧的上下文。

  **输入/输出表示**

>   **为了使BERT处理一系列的下游任务，我们的输入表示能够在一个token序列中明确的表示单个句子和一对句子**。在整个工作中，一个“句子”可能是任意一段连续的文本，而不是实际语言的句子。一个“句子”指的是输入BERT的token序列，这个序列可以是单个序列或者是两个序列连在一起。  
>   我们使用带有30000个token的词汇表做WordPiece嵌入，每个序列的第一个token都是一个特殊的分类符号（[CLS]）。与该token相对应的最终隐藏状态用作分类任务的合计序列表示。句子对打包在一起成为单个序列，我们有两种方式区分句子。第一我们使用特殊的token（[SEP]）将句子分开，其次，我们在每个标记中加入一个学习嵌入，表明它是属于句子A还是句子B。如图1所示，我们用E表示输入嵌入，特殊token[CLS]的最终的隐藏向量为C ∈ RH，和 ith 输入token最终隐藏向量为Ti∈ RH。  
>   对于给定的token，其输入表示通过对相应的token，段和位置嵌入求和来构造。这种结构的可视化可以在图2中看到。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007214942271.png#pic_center)

### 3.1 预训练BERT

  我们没有使用从左到右或者从右到左的语言模型来训练BERT。相反，我们使用本节中描述的**两个无监督**的任务来预训练BERT。该步骤在图1的左侧部分中给出。

  **任务1：Masked LM**

>   直观的说，我们有理由相信**深度双向模型**比一个从左到右模型或是一个浅层的左到右或右到左的模型更加强大。**因为双向调节会允许每个单词间接地“see itself”，并且模型可以在多层次的语境中对目标词进行简单的预测**。  
>   为了训练深度双向的表示，我们只是随机的屏蔽一定百分比的输入token，然后去预测那些mask 掉的token。我们将这个过程称为是“masked LM”（MLM），尽管它在文献中通常被称为完形任务。在这种情况下，对应于mask token的最终的隐藏向量通过词汇表输出softmax，如标准的LM。在我们所有的实验中，我们随机屏蔽每个序列中所有的WordPiece token的15%。和去噪的auto-encoder相比，我们只预测掩蔽的单词而不是重建整个输入。  
>   虽然这允许我们获得**双向预训练模型**，但**缺点是我们在预训练和微调之间产生不匹配，因为微调期间 [MASK] 不会出现**。为了缓解这个问题，我们并不总是用实际的 [MASK] token替换“masked”词。训练数据生成器随机选择15％的token位置进行预测。如果选择了第i个token，我们将第i个token （1）80%的情况替换为[MASK] token（2）10%的情况替换为随机的token（3）10%的情况保持不变。然后Ti 将用于预测具有交叉熵损失的原始token。我们比较了附录C.2中该过程的变化。

  **任务2： 下一句话预测（NSP）**

>   许多重要的下游任务，例如问答（QA）和自然语言推理（NLI），都是基于理解两个句子之间的关系，而不是由语言建模直接捕获的。**为了训练理解句子关系的模型**，我们预训练了二进制的下一句预测任务，该任务可以从任何单语语料库中轻松的生成。具体而言，当为每个预训练例子选择句子A和B时，50％的情况下，B是A后面的下一个句子（**标记为IsNext**），并且50％的情况，B是来自语料库的随机句子（**标记为作为NotNext**）。正如我们在图1中所示，C用于下一句话预测（NSP)，尽管它很简单，但我们在5.1节中演示了对这项任务的预训练，对QA和NLI都是非常有益的。NSP任务与Jernite等人使用的表征学习目标密切相关。 然而在以前的工作中，只有句子嵌入被转移到下游任务，其中BERT传输所有参数以初始化最终任务模型参数。

  **预训练数据**

>   预训练过程很大程度上遵循了有关语言模型预训练的现有文献。我们使用 **BooksCorpus（8 亿个词）和英文维基百科（25 亿个词）**。对于维基百科，我们只抽取文本段落，舍弃了列表、表格和头部。为了提取长连续序列，关键是要使用**文档级的语料库**，而不是句子级的语料（如 Billion Word Benchmark），以获取长的近邻序列。

### 3.2 微调BERT

  微调简单直接的，**因为Transformer中的自注意力机制允许BERT通过交换适当的输入和输出来模拟许多下游任务——无论它们是单文本还是文本对**。对于涉及文本对的应用，常见的模式是在应用双向交叉注意力之前独立编码文本对，例如Parikh et al. (2016); Seo et al. (2017)。BERT使用self-attention机制来统一这两个阶段，因为编码具有自注意力的连接文本对有效地包括两个句子之间的双向交叉注意力。

  对于每个任务，我们**只需将任务特定的输入和输出插入到BERT中，并端对端微调所有参数**。在输入中，来自预训练的句子A和句子B与 （1）释义中的句子对，（2）假设中的 hypothesis-premise 对，（3）问答中的疑问句对，以及（4）文本分类或序列标记中退化text-∅对。在输出中，token表示被馈送到用于符号级任务的输出层，例如序列标记或问答，并且[CLS]表示被馈送到输出层以用于分类，例如蕴含关系或是情感分析。

  与预训练相比，**微调**相对容易。从完全相同的预训练模型开始，本文中的所有结果可以在**单个Cloud TPU上最多1小时复现，或者在GPU上几小时复制**。在第四节相应的小节中，我们描述了任务特定的细节。更多细节可以在附录A.5中找到。

4 实验
----

  在本节中，我们将介绍**11个NLP任务的BERT微调结果**。

### 4.1 GLUE

  通用语言理解评估（GLUE)基准是各种自然语言理解任务的集合。GLUE数据集的详细说明见附录B.1。

  为了在GLUE上微调，我们按照第3节中描述的输入序列（单个句子或句子对），并使用与第一个输入符号([CLS])对应的最终隐藏向量C∈RH作为聚合表示。在微调期间引入的唯一新参数是分类层权重W∈RK×H，其中K是标签数。我们用C和W计算标准分类损失，即 log（softmax（CWT））。

  对于所有GLUE任务，我们使用**批量大小为32并对3个epochs的数据进行微调**。对于每项任务，我们在开发集上选择了最佳的微调学习率（5e-5,4e-5,3e-5和2e-5）。此外，对于BERTLARGE，我们发现微调有时在小数据集上不稳定，因此我们进行了**几次随机重启**，并在开发集上选择了最佳模型。通过随机重启，我们**使用相同的预训练的检查点，但执行不同的微调数据和分类层的初始化**。

  结果如表1所示。BERTBASE和BERTLARGE在所有任务上都大大优于所有任务，与现有技术相比，平均准确度提高了4.5％和7.0％。请注意，除了注意力masking之外，BERTBASE和OpenAI GPT在模型架构方面几乎完全相同。对于最大和最广泛报告的GLUE任务，MNLI、BERT获得4.6％的绝对准确度提高。在官方GLUE排行榜中，截至撰写之日，BERTLARGE得分为80.5，而OpenAI GPT获得72.8分。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020100721501540.png#pic_center)

  我们发现BERTLARGE在所有任务中都明显优于BERTBASE，特别是那些训练数据非常少的任务。第5.2节将更详细地探讨模型大小的影响。

### 4.2 SQuAD v1.1

  斯坦福问答数据集（SQuAD v1.1） **是一个有10万问答对的集合。给定一个问题以及Wikipedia中包含答案的段落，任务是预测段落中的答案文本范围**。

  如图1所示，在**问答任务**中，我们将输入的问题和段落表示为一个单独压缩序列，问题使用A嵌入，而段落使用B嵌入。在**微调**期间，我们只引入了一个起始向量S∈RH和一个结束向量E∈RH。单词 i 作为答案开始的概率，计算Ti和S之间的点积，然后是段落中所有单词的softmax：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215026695.png#pic_center)

。类似的公式用于答案的末端。从位置i到位置j的候选跨度的得分被定义为S·Ti + E·Tj，以及 j ≥ i 用作预测的最大得分跨度。训练目标是正确的开始和结束位置的概率的总和。我们进行3个epoch，学习率为5e-5，批处理大小为32。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215033888.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)

  表2显示了顶级排行榜条目以及来顶级发布系统的结果。**SQuAD排行榜**的最高结果并没有最新的公共系统描述，并且在训练他们的系统时可以使用任何公共数据。因此，在SQUAD上进行微调之前，我们通过**TriviaQA的第一次微调**，在我们的系统中使用适度的数据增强。

  我们表现最佳的系统在整体排名中优于顶级排行榜系统+1.5 F1，在单一系统中优于顶级排行榜系统+1.3 F1。事实上，我们的**单一BERT模型**在F1得分方面优于顶级全体系统。如果没有TriviaQA微调数据，我们只会损失0.1-0.4 F1，仍然大幅超越所有现有系统。

### 4.3 SQuAD v2.0

  SQuAD 2.0任务**通过允许在所提供的段落中没有简短答案的可能性扩展了SQuAD 1.1问题的定义，从而使问题更加实际**。

  对此任务我们使用一种简单的方法来**扩展的SQuAD v1.1 BERT模型**。我们将没有答案的问题视为具有从[CLS]符号处开始和结束的答案范围。开始和结束答案范围位置的概率空间被扩展为包括[CLS]符号的位置。对于预测，我们将无应答范围：snull = S·C + E·C 的得分与最佳非零范围  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215047411.png#pic_center)

的得分进行比较。我们在  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215053294.png#pic_center)

中预测非空答案，其中在开发集上选择阈值τ以最大化F1。我们没有此模型上使用**TriviaQA数据**。我们微调了2个epoch，学习率为5e-5，batch size为48。

  结果与之前的排行榜条目和顶级出版作品相比较，如表3所示，不包括使用BERT作为其组件之一的系统。我们观察到比之前的最佳系统提高了+5.1 F1。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215059992.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)

### 4.4 SWAG

  具有对抗性生成的情况（SWAG）数据集包含113k个句子对完成示例，其评估基于常识的推论。给出一个句子，任务是在四个选择中选择最合理的继续。

  当对SWAG数据集进行微调时，我们构造了**四个输入序列**，每个输入序列包含给定句子（句子A）的连接和可能的继续（句子B）。引入的唯一任务特定参数是**向量**，其与 [CLS] token的表示C的点积，表示每个选择的得分，并使用softmax层对其进行归一化。

  我们**微调模型**，使用3个epoch，学习率为2e-5，batch size为16。结果如表4所示。BERTLARGE的比作者的基线ESIM + ELMo系统高+ 27.1％，比OpenAI GPT 高8.3 ％。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215110634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)

5 消融研究
------

  尽管我们已经展示了非常强有力的实证结果，但是到目前为止展示的结果并没有从BERT框架的各个方面分离出具体的贡献。在本节中，我们**对BERT的多个方面进行消融实验**，以便更好地理解它们的相对重要性。

### 5.1 预训练任务的效果

  我们的核心主张之一是，**BERT的深层双向性是BERT与以前的工作相比最重要的改进，这是通过masked LM预训练实现的**。为了证明这一说法，我们评估了两种新模型，它们使用与BERTBASE完全相同的预训练数据、微调方案和transformer超参数:

> *   No NSP: **使用“masked LM”（MLM）训练但没有“下一句预测”（NSP）任务的模型**。
> *   LTR & No NSP: **使用Left-to-Right（LTR）LM而不是MLM训练的模型。在这种情况下，我们预测每个输入字，不应用任何masked**。 左边的约束也适用于微调，因为我们发现使用双向上下文进行预训练并且使用双向上下文进行微调总是更糟。此外，该模型在没有NSP任务的情况下进行了预训练。 这与OpenAI GPT直接相当，但使用我们更大的训练数据集，输入表示和我们的微调方案。

  结果见表5。我们首先说明NSP任务带来的影响。我们可以看到，去除NSP会严重影响**QNLI、MNLI和SQuAD**的表现。这些结果表明，我们的预训练方法对于获得之前给出的强有力的实证结果至关重要。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215146870.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)

  接下来，我们通过比较“No NSP”和“LTR & No NSP”来评估训练双向表达的影响。LTR模型在所有任务上的表现都比MLM模型差，MRPC和SQuAD的下降幅度非常大。对于SQuAD来说，直觉上很清楚，LTR模型在跨度和词汇预测方面表现非常差，因为词级 masked 状态没有右侧上下文。因为MRPC不清楚性能差是由于数据量小还是任务的性质，但是我们发现这种性能差在多次随机重启的超参数扫描中是一致的。

  为了尝试增强LTR系统，我们尝试**在LTR系统之上添加一个运行初始化的BiLSTM**，以进行微调。这确实大大提高了小队的成绩，但是成绩仍然比不上预训练的双向模型。

  我们认识到，也可以训练单独的LTR和RTL模型，并像ELMo一样，将每个词表示为两个模型的拼接。然而: **( a )训练成本是一个双向模型的两倍；( b )对于像QA这样的任务来说，这是不直观的，因为RTL模型无法对这个问题的答案进行限定；( c )这严格来说不如深度双向模型强大，因为深度双向模型可以选择使用左语境或右语境**。

### 5.2 模型大小的影响

  在本节中，我们**探讨模型大小对微调任务准确性的影响**。我们训练了许多BERT模型，这些模型具有不同的层数、隐藏单元和注意head数，而另外使用了与前面描述的相同的超参数和训练过程。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215200532.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)

  表6显示了所选**GLE任务**的结果。在此表中，我们报告了5次随机微调重启的平均Dev Set精度。我们可以看到，更大的模型可以在所有四个数据集上实现严格的精度改进，即使对于MRPC来说也是如此，因为MRPC只有3600个词的训练示例，与训练前的任务有很大不同。也许令人惊讶的是，我们能够在相对于现有文献来说已经准备好的模型之上实现如此显著的改进。最近出现的最大的transformer是Vaswani（2017）探索的 (L=6, H=1024, A=16)，encoder参数有100m。我们发现在文献中看见最大的模型是AI-Rfou（2018）(L=64, H=512, A=2)一共235m参数。BERTBASE有110m参数，BERTLARGE有340m参数。

  多年来，人们已经知道，**增加模型大小将导致大规模任务的持续改进**，如机器翻译和语言建模，这一点从表6所示的长期训练数据的LM复杂度中可以看出。然而，我们认为，这是第一次证明，如果模型经过充分的预训练，缩小到极端的模型大小也会导致非常小规模任务的大幅度改进。

### 5.3 基于特征的BERT方法

  到目前为止呈现的所有BERT结果都使用了**微调方法**，在该方法中，一个简单的分类层被添加到预先训练的模型中，并且所有参数都在下游任务中被联合微调。然而，基于特征的方法具有某些优势，在这种方法中，固定维度的词向量将被提取出来。**首先，并非所有NLP任务都可以由Transformer编码器架构轻松表示，因此需要添加特定于任务的模型架构。第二，能够预计算一次训练数据的表示方法，然后在此表示的基础上用较简单的模型进行许多实验，这有很大的计算优势**。

  在本节中，我们通过在**CoNLL-2003 NER任务**上生成类似ELMo的预训练的语境表示，来评估基于特征的方法中BERT每种形式的表现。为此，我们使用与第4.3节相同的输入表示，但是**使用一个或多个层的激活，不微调BERT的任何参数**。在分类层之前，这些上下文嵌入被用作运行初始化的两层768维BiLSTM的输入。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201007215217325.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)

  结果显示在表7中。最佳的执行方法是将预训练好的Transformer的前四个隐藏层中的词表示拼接起来，这仅比微调整个模型落后0.3 F1。这表明**BERT对于微调和基于特征的方法都是有效的**。

6 结论
----

  由于语言模型的迁移学习，最近的研究表明，**丰富的、无监督的预训练**是许多语言理解系统的一个组成部分。特别是，这些结果使得即使是低资源的任务也能从深度单向架构中受益。我们的**主要贡献是将这些发现进一步推广到更深度双向架构中，允许相同的预训练模型成功地处理一系列广泛的NLP任务**。

附录
--

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008095554130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008095615104.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008095631428.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201008095645405.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjY5MTU4NQ==,size_16,color_FFFFFF,t_70#pic_center)