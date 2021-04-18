# Deep Learning

## 过拟合(overfitting)

**过拟合：**在训练数据上的误差非常小，而在测试数据上误差反而增大。

解决过拟合的方法：
1、增加数据量
2、选用shallow的网络模型降低模型复杂度
3、正则化(regularization)：对模型参数添加先验，使得模型复杂度较小，对于噪声以及outliers的输入扰动相对较小
4、加dropout
5、对数据做数据增强(data augumentation)
6、early stopping

**动量(momentum)：**将上一次的梯度变化考虑在内，提高收敛的速度

