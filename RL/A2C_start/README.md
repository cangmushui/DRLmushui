## A2C

链接

https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f





## 代码阅读

1.设置随机数种子保证可复现

2.1)if else三元运算符  2)a=[i+1 for i in range(4)] 循环返回列表 3)反向迭代 for n in reversed(迭代器)

3.想要使用GPU要先把数据和模型都放到GPU上

!4.怎样新建一个子进程环境序列

5.对随机策略进行采样的固定代码

```python
probs = policy_network(state)
# Note that this is equivalent to what used to be called multinomial
m = Categorical(probs)
action = m.sample()
next_state, reward = env.step(action)
loss = -m.log_prob(action) * reward
loss.backward()
```







A2C算法训练步骤:

1设置8个子环境

2新建模型,把模型放到device上,新建Adam优化器,把模型参数放进优化器

3重置环境,当小于最大训练步数时,一直循环:

 - 5次一轮,每轮执行(暂时忽略entropy是什么)

   1)将state变为tensor并放到device上,输入到网络,得到随机策略,并进行采样得到action,执行step函数得到next_state, reward, done, _

   2)求出所选动作的概率的log值,放到log_probs列表中,此外还有,状态价值列表values,rewards列表(放到device上),done_mask列表(放到device上),一轮过后这些列表都是包含5个元素,每个元素都是一个8x1的tensor

   3)把next_state赋值给state

- 把next_state转化成tensor并放到device上

- 将next输入网络得到next_value(8x1tensor)

- returns = compute_returns(next_value, rewards, masks),计算returns,就是公式里面r+nextvalue那一部分,

  returns仍然是一个长度为5的列表,每个元素都是8x1tensor,前面放在device是为了这一步方便计算returns,个人觉得这块的实现有问题,直接r+nextValue就好了.

- 将log_probs,returns,values三个列表都cat成为40X1的tensor并且阻断(detach)returns的梯度传播
- 写损失函数,进行梯度下降,并将该阻断的部分阻断,这里将Actor损失函数和Critic损失函数加在一起梯度下降



