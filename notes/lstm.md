遗忘门，输入门，输出门
https://zhuanlan.zhihu.com/p/518848475

1.遗忘门
Ft = sigmoid(w[Ht-1 ,Xt]+B)

2.输入门
C't = tanh(w[Ht-1 ,Xt]+B)
It = sigmoid(w[Ht-1 ,Xt]+B)

Ct = Ct-1 * Ft + C't * It

3.输出门
Ot = sigmoid(w[Ht-1 ,Xt]+B)
Ht = Ot * tanh(Ct)