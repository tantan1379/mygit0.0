# Deep Learning（pytorch)

## Basic knowledge

* tensor比ndarray的优势在于可以使用GPU进行加速计算

* 有多少个卷积核就有多少个feature map，一个feature map对应图像被提取的一种特征

* output = ( input - K + 2 * P ) / S + 1 ，改变S可以改变输入的维度

* 经过trainloader加载后，loader的维度为(batchsize,channel,height,width)

* DoubleTensor比FloatTensor有更高的精度，适合增强学习

------

## Tensor

**Convert**

1、cpu –> gpu：`data.cuda()`

2、gpu –> cpu：`data.cpu()`

3、Numpy.ndarray –> Tensor **（导入）**： `torch.from_numpy(data)`

4、Tensor –> Numpy.ndarray ：`data.numpy()`

5、Tensor -> DoubleTensor: `torch.set_default_tensor_tepe(torch.DoubleTensor)`

6、将List转换为Tensor，生成单精度浮点类型的张量：`torch.Tensor(data)` 同torch.FloatTensor()

7、根据原始数据类型生成相应的张量：`torch.tensor(data)`

8、将tensor转换为python对象类型：`a.item()`：对只含一个元素的tensor使用，,

**Create tensor**

1、随机创建指定形状的Tensor：`torch.Tensor(*sizes)`

2、生成从s到e(不包含e)，步长为step的一维Tensor ：`torch.arange(s, e, step)`
生成从s到e(包含e)，元素个数为steps的一维Tensor：`torch.linspace(s,e,steps)`

3、生成随机分布的Tensor:
标准分布(0,1正态分布) ：`torch.randn(*size)`
均匀分布：`torch.rand(*size)`

4、创建特殊的Tensor：`torch.ones(*size)` `torch.zeros(*size)` `torch.eyes(*size)`

5、创建具有相同值的Tensor:`torch.full(*size,val)` 如果size写[]，生成标量

**Index&slice**

1、选取指定维度进行切片：`a.index_select(dim,torch.tensor)`  2、冒号    3、省略号 

**Dimension**

1、调整Tensor的形状（常用）：`tensor.view()` e.g.:`x=x.view(x.size(0),-1)`
**notes:** 在神经网络中图像的维度为(batchsize,channel,height,width),一定要以这个顺序和逻辑进行view；view前后的size要相同

2、修改维度
（增维）：`tensor.unsqueeze(pos)` 在posi前的一个位置加一维
（减维）：`tensor.squeeze()`自动挤压所有值为1的维度 `tensor.squeeze(pos)`挤压（减去）pos位置的维度
（维度扩展）：`tensor.expand(*size)` [=] 
**note:** 1、自动复制broadcast tensor，变换前后维数不变；  2、需要扩张的维值必须为1；  3、如果某一位置填-1则表示该维保持不变

3、交换维度：
（二维）：`tensor.t()`
（多维）：`tensor.transpose(dim1,dim2)`
（通用）：`tensor.permute(dim1,dim2,dim3,...)`

4、Broadcast：
先在某一维度**之前**插入维度(unsqueeze)，再在大小为1的维度上进行扩张(expand)

5、topk:

`torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)`

**Merge&Split**

1、cat:`torch.cat([a,b],dim) `两个拼接的tensor必须在dim维之外的维度均相等

**Property**

查看Tensor的大小：`tensor.size()` 
查看Tensor的大小：`tensor.shape` 
统计Tensor的元素个数：`Tensor.numel()`   

**Comparision**

(1) tensor和Tensor(FloatTensor)的区别在于，tensor只能接受现有的数据，Tensor可以接受数据的维度()或数据([])
为避免混淆，使用时建议，要维度用大写Tensor，要具体数据用小写tensor

------

## Numpy 
**（以下用np表示）**
1、`data.numpy()` 
将tensor类型转换为numpy类型; 
2、`np.equal(x1, x2)` 
Return (x1 == x2) element-wise. 
比较两个数组的值(若相等,在对应位置上取True，不等取False)；
3、`np.all()`Test whether all array elements along a given axis evaluate to True.
4、`if array`判断numpy数组是否为空，将列表作为布尔值，若不为空返回True，否则视为False;

---
## torchvision
1、在datasets模块中保存着各类数据集
2、在models模块中保存搭建好的网络（可以不加载数据）
3、在transforms模块中封装了一些处理数据的方法
**Note:** 
1.torchvision的datasets的输出是[0,1]的PILImage，所以我们需要归一化为[-1,1]的Tensor
2.数据增强虽然会使训练过程收敛变慢，但可以提高测试集准确度，防止过拟合，提高模型的泛化能力。 

---

## dataloader

参数表：

```python
class torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=<function default_collate>,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None)
```

shuffle：设置为True的时候，每个epoch都会打乱数据集 
collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能 
drop_last：告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留

---

## Pandas

Pandas的基础结构可以分为两种：**数据框和序列**。

**数据框（DataFrame）**是拥有轴标签的二维链表，换言之数据框是拥有标签的行和列组成的矩阵 - 列标签位列名，行标签为索引。Pandas中的行和列是Pandas序列 - 拥有轴标签的一维链表。

`iterrows()` 是在数据框中的行进行迭代的一个生成器，它返回每行的索引及一个包含行本身的对象。

---



## Others

一、`torch.backends.cudnn.benchmark = True`
大部分情况下，设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题

---

二、`sum(p.numel() for p in model.parameters() if p.requires_grad)` 计算网络的参数量