TFIDF  
理解  
为了分析某个词语在文章中的权重  
用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降  

TF(term frequency)：  
词在文章中出现的频率。它的计算要依据个人的情况而定，只要保证这个TF能表示词在文章的频率就行  
<div align="center"><img src="../../assets/tfidf1.png"></div>  

IDF(inverse document frequency):  
IDF = 文章总数/包含该词的文档数  
<div align="center"><img src="../../assets/tfidf2.png"></div>  

<div align="center"><img src="../../assets/tfidf3.png"></div>  
