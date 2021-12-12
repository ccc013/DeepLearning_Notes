# 基础
## 张量 Tensor
参考文章：

1. [4-1,张量的结构操作](https://github.com/lyhue1991/eat_pytorch_in_20_days/blob/master/4-1,%E5%BC%A0%E9%87%8F%E7%9A%84%E7%BB%93%E6%9E%84%E6%93%8D%E4%BD%9C.md)

​

张量的操作主要分结构操作和数学运算操作两种：

1. 结构操作：创建、索引切片、维度变换、合并分割等
1. 数学运算：标量运算、向量运算、矩阵运算等。

​

### 结构操作
#### 创建张量
创建张量和 numpy 创建数组方法比较相似，代码示例如下，总结一下大概可以这么生成：

1. torch.tensor()：直接输入向生成 tensor 的数值；
1. torch.arange() 或者 torch.linspace()：指定一个范围以及步长
1. 生成特定数值的矩阵：torch.ones(), torch.zeros(), torch.eyes(),torch.diag()
1. torch.fill_()：改变一个 tensor 的数值
1. 生成指定的数据分布的 tensor：torch.randperm, torch.normal, 

​

```python
import numpy as np
import torch 

a = torch.tensor([1,2,3],dtype = torch.float)
print(a)
# tensor([1., 2., 3.])

b = torch.arange(1,10,step = 2)
print(b)
# tensor([1, 3, 5, 7, 9])

c = torch.linspace(0.0,2*3.14,10)
print(c)
# tensor([0.0000, 0.6978, 1.3956, 2.0933, 2.7911, 3.4889, 4.1867, 4.8844, 5.5822, 6.2800])

d = torch.zeros((3,3))
print(d)
# tensor([[0., 0., 0.],
#        [0., 0., 0.],
#        [0., 0., 0.]])

a = torch.ones((3,3),dtype = torch.int)
b = torch.zeros_like(a,dtype = torch.float)
print(a)
print(b)
# tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int32)
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])

torch.fill_(b,5)
print(b)
# tensor([[5., 5., 5.],
        [5., 5., 5.],
        [5., 5., 5.]])

#均匀随机分布
torch.manual_seed(0)
minval,maxval = 0,10
a = minval + (maxval-minval)*torch.rand([5])
print(a)
# tensor([4.9626, 7.6822, 0.8848, 1.3203, 3.0742])

#正态分布随机
b = torch.normal(mean = torch.zeros(3,3), std = torch.ones(3,3))

#正态分布随机
mean,std = 2,5
c = std*torch.randn((3,3))+mean

#整数随机排列
d = torch.randperm(20)

#特殊矩阵
I = torch.eye(3,3) #单位矩阵
print(I)
t = torch.diag(torch.tensor([1,2,3])) #对角矩阵
```


#### 索引切片
张量的索引切片方式和numpy几乎是一样的。切片时支持缺省参数和省略号。
可以通过索引和切片对部分元素进行修改。
如果要通过修改张量的某些元素得到新的张量，可以使用`torch.where,torch.masked_fill,torch.index_fill`
​

代码示例：
```python
#均匀随机分布
torch.manual_seed(0)
minval,maxval = 0,10
t = torch.floor(minval + (maxval-minval)*torch.rand([5,5])).int()
print(t)

# tensor([[4, 7, 0, 1, 3],
        [6, 4, 8, 4, 6],
        [3, 4, 0, 1, 2],
        [5, 6, 8, 1, 2],
        [6, 9, 3, 8, 4]], dtype=torch.int32)
#第0行
print(t[0])
# tensor([4, 7, 0, 1, 3], dtype=torch.int32)

#倒数第一行
print(t[-1])

#第1行第3列
print(t[1,3])
print(t[1][3])
# tensor(4, dtype=torch.int32)
# tensor(4, dtype=torch.int32)

#第1行至第3行
print(t[1:4,:])

#第1行至最后一行，第0列到最后一列每隔两列取一列
print(t[1:4,:4:2])

#可以使用索引和切片修改部分元素
x = torch.tensor([[1,2],[3,4]],dtype = torch.float32,requires_grad=True)
x.data[1,:] = torch.tensor([0.0,0.0])
# 输出 x：tensor([[1., 2.],
#        [0., 0.]], requires_grad=True)

a = torch.arange(27).view(3,3,3)

#省略号可以表示多个冒号，输出第二列
print(a[...,1])
```
接下来是对于不规则切片的提取，可以使用`torch.index_select, torch.masked_select, torch.take`。
假设一个班级成绩册的例子，有4个班级，每个班级10个学生，每个学生7门科目成绩。可以用一个4×10×7的张量来表示。
```python
minval=0
maxval=100
scores = torch.floor(minval + (maxval-minval)*torch.rand([4,10,7])).int()
print(scores)

# tensor([[[55, 95,  3, 18, 37, 30, 93],
         [17, 26, 15,  3, 20, 92, 72],
         [74, 52, 24, 58,  3, 13, 24],
         [81, 79, 27, 48, 81, 99, 69],
         [56, 83, 20, 59, 11, 15, 24],
         [72, 70, 20, 65, 77, 43, 51],
         [61, 81, 98, 11, 31, 69, 91],
         [93, 94, 59,  6, 54, 18,  3],
         [94, 88,  0, 59, 41, 41, 27],
         [69, 20, 68, 75, 85, 68,  0]],

        [[17, 74, 60, 10, 21, 97, 83],
         [28, 37,  2, 49, 12, 11, 47],
         [57, 29, 79, 19, 95, 84,  7],
         [37, 52, 57, 61, 69, 52, 25],
         [73,  2, 20, 37, 25, 32,  9],
         [39, 60, 17, 47, 85, 44, 51],
         [45, 60, 81, 97, 81, 97, 46],
         [ 5, 26, 84, 49, 25, 11,  3],
         [ 7, 39, 77, 77,  1, 81, 10],
         [39, 29, 40, 40,  5,  6, 42]],

        [[50, 27, 68,  4, 46, 93, 29],
         [95, 68,  4, 81, 44, 27, 89],
         [ 9, 55, 39, 85, 63, 74, 67],
         [37, 39,  8, 77, 89, 84, 14],
         [52, 14, 22, 20, 67, 20, 48],
         [52, 82, 12, 15, 20, 84, 32],
         [92, 68, 56, 49, 40, 56, 38],
         [49, 56, 10, 23, 90,  9, 46],
         [99, 68, 51,  6, 74, 14, 35],
         [33, 42, 50, 91, 56, 94, 80]],

        [[18, 72, 14, 28, 64, 66, 87],
         [33, 50, 75,  1, 86,  8, 50],
         [41, 23, 56, 91, 35, 20, 31],
         [ 0, 72, 25, 16, 21, 78, 76],
         [88, 68, 33, 36, 64, 91, 63],
         [26, 26,  2, 60, 21,  5, 93],
         [17, 44, 64, 51, 16,  9, 89],
         [58, 91, 33, 64, 38, 47, 19],
         [66, 65, 48, 38, 19, 84, 12],
         [70, 33, 25, 58, 24, 61, 59]]], dtype=torch.int32)
        
# 抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))
# tensor([[[55, 95,  3, 18, 37, 30, 93],
         [72, 70, 20, 65, 77, 43, 51],
         [69, 20, 68, 75, 85, 68,  0]],

        [[17, 74, 60, 10, 21, 97, 83],
         [39, 60, 17, 47, 85, 44, 51],
         [39, 29, 40, 40,  5,  6, 42]],

        [[50, 27, 68,  4, 46, 93, 29],
         [52, 82, 12, 15, 20, 84, 32],
         [33, 42, 50, 91, 56, 94, 80]],

        [[18, 72, 14, 28, 64, 66, 87],
         [26, 26,  2, 60, 21,  5, 93],
         [70, 33, 25, 58, 24, 61, 59]]], dtype=torch.int32)

#抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
q = torch.index_select(torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))
                   ,dim=2,index = torch.tensor([1,3,6]))
print(q)
# tensor([[[95, 18, 93],
         [70, 65, 51],
         [20, 75,  0]],

        [[74, 10, 83],
         [60, 47, 51],
         [29, 40, 42]],

        [[27,  4, 29],
         [82, 15, 32],
         [42, 91, 80]],

        [[72, 28, 87],
         [26, 60, 93],
         [33, 58, 59]]], dtype=torch.int32)

# 抽取第0个班级第0个学生的第0门课程，第2个班级的第4个学生的第1门课程，第3个班级的第9个学生第6门课程成绩
#take将输入看成一维数组，输出和index同形状
s = torch.take(scores,torch.tensor([0*10*7+0,2*10*7+4*7+1,3*10*7+9*7+6]))
# <tf.Tensor: shape=(3, 7), dtype=int32, numpy=
array([[52, 82, 66, 55, 17, 86, 14],
       [99, 94, 46, 70,  1, 63, 41],
       [46, 83, 70, 80, 90, 85, 17]], dtype=int32)>

#抽取分数大于等于80分的分数（布尔索引）
#结果是1维张量
g = torch.masked_select(scores,scores>=80)
print(g)

```
以上这些方法仅能提取张量的部分元素值，但不能更改张量的部分元素值得到新的张量。
如果要通过修改张量的部分元素值得到新的张量，可以使用`torch.where,torch.index_fill 和 torch.masked_fill`：

1. `torch.where`可以理解为if的张量版本。
1. `torch.index_fill`的选取元素逻辑和torch.index_select相同。
1. `torch.masked_fill`的选取元素逻辑和torch.masked_select相同。
```python
#如果分数大于60分，赋值成1，否则赋值成0
ifpass = torch.where(scores>60,torch.tensor(1),torch.tensor(0))
print(ifpass)

#将每个班级第0个学生，第5个学生，第9个学生的全部成绩赋值成满分
torch.index_fill(scores,dim = 1,index = torch.tensor([0,5,9]),value = 100)
#等价于 scores.index_fill(dim = 1,index = torch.tensor([0,5,9]),value = 100)

#将分数小于60分的分数赋值成60分
b = torch.masked_fill(scores,scores<60,60)
#等价于b = scores.masked_fill(scores<60,60)
```
​

#### 维度变换
维度变换相关函数主要有：

1. `torch.reshape` 可以改变张量的形状。 
1. `torch.squeeze` 可以减少维度。
1. `torch.unsqueeze` 可以增加维度。
1. `torch.transpose` 可以交换维度。



```python
# 张量的view方法有时候会调用失败，可以使用reshape方法。
torch.manual_seed(0)
minval,maxval = 0,255
a = (minval + (maxval-minval)*torch.rand([1,3,3,2])).int()
print(a.shape)
print(a)

# torch.Size([1, 3, 3, 2])
tensor([[[[126, 195],
          [ 22,  33],
          [ 78, 161]],

         [[124, 228],
          [116, 161],
          [ 88, 102]],

         [[  5,  43],
          [ 74, 132],
          [177, 204]]]], dtype=torch.int32)
# 改成 （3,6）形状的张量
b = a.view([3,6]) #torch.reshape(a,[3,6])
print(b.shape)

# torch.Size([3, 6])
tensor([[126, 195,  22,  33,  78, 161],
        [124, 228, 116, 161,  88, 102],
        [  5,  43,  74, 132, 177, 204]], dtype=torch.int32)
# 改回成 [1,3,3,2] 形状的张量
c = torch.reshape(b,[1,3,3,2]) # b.view([1,3,3,2]) 


```
如果张量在某个维度上只有一个元素，利用torch.squeeze可以消除这个维度。
torch.unsqueeze的作用和torch.squeeze的作用相反。
```python
a = torch.tensor([[1.0,2.0]])
s = torch.squeeze(a)
print(a)
print(s)
print(a.shape)
print(s.shape)

# tensor([[1., 2.]])
tensor([1., 2.])
torch.Size([1, 2])
torch.Size([2])

#在第0维插入长度为1的一个维度

d = torch.unsqueeze(s,axis=0)  
print(s)
print(d)

print(s.shape)
print(d.shape)

# tensor([1., 2.])
tensor([[1., 2.]])
torch.Size([2])
torch.Size([1, 2])
```
`torch.transpose`可以交换张量的维度，`torch.transpose`常用于图片存储格式的变换上。
如果是二维的矩阵，通常会调用矩阵的转置方法 `matrix.t()`，等价于 `torch.transpose(matrix,0,1)`。
```python
minval=0
maxval=255
# Batch,Height,Width,Channel
data = torch.floor(minval + (maxval-minval)*torch.rand([100,256,256,4])).int()
print(data.shape)

# 转换成 Pytorch默认的图片格式 Batch,Channel,Height,Width 
# 需要交换两次
data_t = torch.transpose(torch.transpose(data,1,2),1,3)
print(data_t.shape)
# torch.Size([100, 256, 256, 4])
torch.Size([100, 4, 256, 256])

matrix = torch.tensor([[1,2,3],[4,5,6]])
print(matrix)
print(matrix.t()) #等价于torch.transpose(matrix,0,1)

# tensor([[1, 2, 3],
        [4, 5, 6]])
tensor([[1, 4],
        [2, 5],
        [3, 6]])

```
​

#### 合并分割
可以用`torch.cat`方法和`torch.stack`方法将多个张量合并，可以用`torch.split`方法把一个张量分割成多个张量。
`torch.cat`和`torch.stack`有略微的区别，`torch.cat`是连接，不会增加维度，而`torch.stack`是堆叠，会增加维度。
```python
a = torch.tensor([[1.0,2.0],[3.0,4.0]])
b = torch.tensor([[5.0,6.0],[7.0,8.0]])
c = torch.tensor([[9.0,10.0],[11.0,12.0]])

abc_cat = torch.cat([a,b,c],dim = 0)
print(abc_cat.shape)
print(abc_cat)
# torch.Size([6, 2])
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])

abc_stack = torch.stack([a,b,c],axis = 0) #torch中dim和axis参数名可以混用
print(abc_stack.shape)
print(abc_stack)

# torch.Size([3, 2, 2])
tensor([[[ 1.,  2.],
         [ 3.,  4.]],

        [[ 5.,  6.],
         [ 7.,  8.]],

        [[ 9., 10.],
         [11., 12.]]])

#
torch.cat([a,b,c],axis = 1)

# tensor([[ 1.,  2.,  5.,  6.,  9., 10.],
        [ 3.,  4.,  7.,  8., 11., 12.]])

#
torch.stack([a,b,c],axis = 1)

# tensor([[[ 1.,  2.],
         [ 5.,  6.],
         [ 9., 10.]],

        [[ 3.,  4.],
         [ 7.,  8.],
         [11., 12.]]])
```
`torch.split`是`torch.cat`的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割。
```python
print(abc_cat)
a,b,c = torch.split(abc_cat,split_size_or_sections = 2,dim = 0) #每份2个进行分割
print(a)
print(b)
print(c)

# 

print(abc_cat)
p,q,r = torch.split(abc_cat,split_size_or_sections =[4,1,1],dim = 0) #每份分别为[4,1,1]
print(p)
print(q)
print(r)

# tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
tensor([[ 9., 10.]])
tensor([[11., 12.]])
```
​


---

### 数学运算
张量的数学运算符可以分为标量运算符、向量运算符、以及矩阵运算符。
加减乘除乘方，以及三角函数，指数，对数等常见函数，逻辑比较运算符等都是标量运算符。
标量运算符的特点是对张量实施逐元素运算。
有些标量运算符对常用的数学运算符进行了重载。并且支持类似numpy的广播特性。
​

#### 标量运算
标量运算主要包括了加减乘除、**（几次幂）、求模%、整除 //、>=、<=、==、开方 torch.sqrt()、max、min、round、floor、ceil、trunc、fmod、remainder、clamp
```python
import torch 
import numpy as np 

a = torch.tensor([[1.0,2],[-3,4.0]])
b = torch.tensor([[5.0,6],[7.0,8.0]])

x = torch.tensor([2.6,-2.7])
print(torch.round(x)) #保留整数部分，四舍五入
print(torch.floor(x)) #保留整数部分，向下归整
print(torch.ceil(x))  #保留整数部分，向上归整
print(torch.trunc(x)) #保留整数部分，向0归整
# tensor([ 3., -3.])
tensor([ 2., -3.])
tensor([ 3., -2.])
tensor([ 2., -2.])

x = torch.tensor([2.6,-2.7])
print(torch.fmod(x,2)) #作除法取余数 
print(torch.remainder(x,2)) #作除法取剩余的部分，结果恒正
# tensor([ 0.6000, -0.7000])
tensor([0.6000, 1.3000])

# 幅值裁剪
x = torch.tensor([0.9,-0.8,100.0,-20.0,0.7])
y = torch.clamp(x,min=-1,max = 1)
z = torch.clamp(x,max = 1)
print(y)
print(z)

# tensor([ 0.9000, -0.8000,  1.0000, -1.0000,  0.7000])
tensor([  0.9000,  -0.8000,   1.0000, -20.0000,   0.7000])

```


#### 向量运算
向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。
```python
a = torch.arange(1,10).float()
print(torch.sum(a))
print(torch.mean(a))
print(torch.max(a))
print(torch.min(a))
print(torch.prod(a)) #累乘
print(torch.std(a))  #标准差
print(torch.var(a))  #方差
print(torch.median(a)) #中位数
# tensor(45.)
tensor(5.)
tensor(9.)
tensor(1.)
tensor(362880.)
tensor(2.7386)
tensor(7.5000)
tensor(5.)


#指定维度计算统计值
b = a.view(3,3)
print(b)
print(torch.max(b,dim = 0))
print(torch.max(b,dim = 1))

# tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
torch.return_types.max(
values=tensor([7., 8., 9.]),
indices=tensor([2, 2, 2]))
torch.return_types.max(
values=tensor([3., 6., 9.]),
indices=tensor([2, 2, 2]))

#cum扫描
a = torch.arange(1,10)

print(torch.cumsum(a,0))
print(torch.cumprod(a,0))
print(torch.cummax(a,0).values)
print(torch.cummax(a,0).indices)
print(torch.cummin(a,0))

# tensor([ 1,  3,  6, 10, 15, 21, 28, 36, 45])
tensor([     1,      2,      6,     24,    120,    720,   5040,  40320, 362880])
tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
torch.return_types.cummin(
values=tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
indices=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0]))

#torch.sort和torch.topk可以对张量排序
a = torch.tensor([[9,7,8],[1,3,2],[5,6,4]]).float()
print(torch.topk(a,2,dim = 0),"\n")
print(torch.topk(a,2,dim = 1),"\n")
print(torch.sort(a,dim = 1),"\n")

#利用torch.topk可以在Pytorch中实现KNN算法
# torch.return_types.topk(
values=tensor([[9., 7., 8.],
        [5., 6., 4.]]),
indices=tensor([[0, 0, 0],
        [2, 2, 2]]))

torch.return_types.topk(
values=tensor([[9., 8.],
        [3., 2.],
        [6., 5.]]),
indices=tensor([[0, 2],
        [1, 2],
        [1, 0]]))

torch.return_types.sort(
values=tensor([[7., 8., 9.],
        [1., 2., 3.],
        [4., 5., 6.]]),
indices=tensor([[1, 2, 0],
        [0, 2, 1],
        [2, 0, 1]]))

```
​

#### 矩阵运算
矩阵必须是二维的。类似torch.tensor([1,2,3])这样的不是矩阵。
矩阵运算包括：矩阵乘法，矩阵转置，矩阵逆，矩阵求迹，矩阵范数，矩阵行列式，矩阵求特征值，矩阵分解等运算。
```python
#矩阵乘法
a = torch.tensor([[1,2],[3,4]])
b = torch.tensor([[2,0],[0,2]])
print(a@b)  #等价于torch.matmul(a,b) 或 torch.mm(a,b)
# tensor([[2, 4],
        [6, 8]])
#矩阵转置
a = torch.tensor([[1.0,2],[3,4]])
print(a.t())

#tensor([[1., 3.],
        [2., 4.]])
#矩阵逆，必须为浮点类型
a = torch.tensor([[1.0,2],[3,4]])
print(torch.inverse(a))

#tensor([[-2.0000,  1.0000],
        [ 1.5000, -0.5000]])

#矩阵求trace
a = torch.tensor([[1.0,2],[3,4]])
print(torch.trace(a))

#tensor(5.)

#矩阵求范数
a = torch.tensor([[1.0,2],[3,4]])
print(torch.norm(a))

#tensor(5.4772)

#矩阵行列式
a = torch.tensor([[1.0,2],[3,4]])
print(torch.det(a))

# tensor(-2.0000)

#矩阵特征值和特征向量
a = torch.tensor([[1.0,2],[-5,4]],dtype = torch.float)
print(torch.eig(a,eigenvectors=True))

#两个特征值分别是 -2.5+2.7839j, 2.5-2.7839j 

#torch.return_types.eig(
eigenvalues=tensor([[ 2.5000,  2.7839],
        [ 2.5000, -2.7839]]),
eigenvectors=tensor([[ 0.2535, -0.4706],
        [ 0.8452,  0.0000]]))

#矩阵QR分解, 将一个方阵分解为一个正交矩阵q和上三角矩阵r
#QR分解实际上是对矩阵a实施Schmidt正交化得到q

a  = torch.tensor([[1.0,2.0],[3.0,4.0]])
q,r = torch.qr(a)
print(q,"\n")
print(r,"\n")
print(q@r)

#矩阵svd分解
#svd分解可以将任意一个矩阵分解为一个正交矩阵u,一个对角阵s和一个正交矩阵v.t()的乘积
#svd常用于矩阵压缩和降维
a=torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]])

u,s,v = torch.svd(a)

print(u,"\n")
print(s,"\n")
print(v,"\n")

print(u@torch.diag(s)@v.t())

#利用svd分解可以在Pytorch中实现主成分分析降维
tensor([[-0.2298,  0.8835],
        [-0.5247,  0.2408],
        [-0.8196, -0.4019]]) 

tensor([9.5255, 0.5143]) 

tensor([[-0.6196, -0.7849],
        [-0.7849,  0.6196]]) 

tensor([[1.0000, 2.0000],
        [3.0000, 4.0000],
        [5.0000, 6.0000]])
```


#### 广播机制
Pytorch的广播规则和numpy是一样的:

- 如果张量的维度不同，**将维度较小的张量进行扩展**，直到两个张量的维度都一样。
- 如果两个张量**在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为1，那么我们就说这两个张量在该维度上是相容的**。
- 如果两个张量在所有维度上都是相容的，它们就能使用广播。
- 广播之后，**每个维度的长度将取两个张量在该维度长度的较大值**。
- 在任何一个维度上，如果一个张量的长度为1，另一个张量长度大于1，那么在该维度上，就好像是对第一个张量进行了复制。

`torch.broadcast_tensors`可以**将多个张量根据广播规则转换成相同的维度**。


```python
a = torch.tensor([1,2,3])
b = torch.tensor([[0,0,0],[1,1,1],[2,2,2]])
print(b + a) 
# tensor([[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]])

a_broad,b_broad = torch.broadcast_tensors(a,b)
print(a_broad,"\n")
print(b_broad,"\n")
print(a_broad + b_broad)

#tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]]) 

tensor([[0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]]) 

tensor([[1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]])
```



---

## 类型转换


参考文章：


1. [Pytorch中的variable, tensor与numpy相互转化的方法](https://blog.csdn.net/pengge0433/article/details/79459679)



加载模块：


```python
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
```


1. numpy 矩阵和 Tensor 张量的互换



```python
# 将 numpy 矩阵转换为 Tensor 张量
sub_ts = torch.from_numpy(sub_img)   #sub_img为numpy类型

# 将 Tensor 张量转化为 numpy 矩阵 
sub_np1 = sub_ts.numpy()             #sub_ts为tensor张量
```


2. numpy 矩阵和 Variable 互换



```python
# numpy 矩阵转换为 Variable
sub_va = Variable(torch.from_numpy(sub_img))

# 将 Variable 张量转化为 numpy
sub_np2 = sub_va.data.cpu().numpy()
```


3. Tensor 张量和 Variable 互换



```python
# Tensor 转换为 Variable
sub_va = tensor2var(sub_ts)
```

---

## PyTorch 中 backward() 详解


参考：


- [PyTorch 的 backward 为什么有一个 grad_variables 参数？](https://zhuanlan.zhihu.com/p/29923090)
- [详解Pytorch 自动微分里的（vector-Jacobian product）](https://zhuanlan.zhihu.com/p/65609544)
- [PyTorch 中 backward() 详解](https://www.pytorchtutorial.com/pytorch-backward/)

​

​


---

## PyTorch中的钩子（Hook）


参考：


- [pytorch中的钩子（Hook）有何作用](https://www.zhihu.com/question/61044004)
- [Pytorch中autograd以及hook函数详解](https://oldpan.me/archives/pytorch-autograd-hook)



非常形象的定义：(pytorch中的钩子（Hook）有何作用？ - 马索萌的回答 - 知乎 [https://www.zhihu.com/question/61044004/answer/294829738](https://www.zhihu.com/question/61044004/answer/294829738))


> 相当于插件。可以实现一些额外的功能，而又不用修改主体代码。把这些额外功能实现了挂在主代码上，所以叫钩子，很形象。



## 分布式数据并行DistributedDataParallel（DDP）


参考文章：


- [pytorch分布式数据并行DistributedDataParallel（DDP）](https://zhuanlan.zhihu.com/p/107139605)



### 简介


DistributedDataParallel（DDP）在module级别实现数据并行性。它使用[torch.distributed](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/intermediate/dist_tuto.html)包communication collectives来同步梯度，参数和缓冲区。并行性在单个进程内部和跨进程均有用。在一个进程中，DDP将input module 复制到 device_ids 指定的设备，相应地按 batch 维度分别扔进模型，并将输出收集到output_device，这与[DataParallel](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)相似。Across processes, DDP inserts necessary parameter synchronizations in forward passes and gradient synchronizations in backward passes. It is up to users to map processes to available resources, as long as processes do not share GPU devices.


**推荐（通常是最快的方法）为每个module 副本创建一个进程，即在一个进程中不进行任何module复制**。


### `DataParallel` 和 `DistributedDataParallel` 区别


`DataParallel` 和 `DistributedDataParallel` 区别：


1. 如果模型太大而无法容纳在单个GPU上，则必须使用 **model parallel** 将其拆分到多个GPU中。 DistributedDataParallel与模型并行工作； DataParallel目前不提供。
1. **DataParallel是单进程，多线程，并且只能在单台计算机上运行**，而DistributedDataParallel是**多进程**，并且可以在**单机和分布式训练**中使用。因此，即使在单机训练中，您的数据足够小以适合单机，DistributedDataParallel仍要比DataParallel更快。
1. DistributedDataParallel还可以**预先复制模型**，而不是在每次迭代时复制模型，并且可以避免PIL全局解释器锁定。
1. 如果数据和模型同时很大而无法用一个GPU训练，则可以将model parallel（与DistributedDataParallel结合使用。在这种情况下，每个DistributedDataParallel进程都可以model parallel，并且所有进程共同用数据并行



### 基本使用


```python
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()
```


现在，创建一个toy model，将其与DDP封装在一起，并提供一些虚拟输入数据。**请注意，如果训练从随机参数开始，则可能要确保所有DDP进程都使用相同的初始值。否则，全局梯度同步将没有意义。**


```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```


DDP封装了 lower level distributed communication details，并提供了干净的API，就好像它是本地模型一样。对于基本用例，DDP仅需要几个LoCs来设置 process group。在将DDP应用到更高级的用例时，需要注意一些警告。


#### 处理速度不同步时


在DDP中，Model, forward method 和 differentiation of the outputs是分布式的同步点。期望不同的过程以相同的顺序到达同步点，并在大致相同的时间进入每个同步点。否则，快速流程可能会提早到达，并在等待时超时。因此，用户负责进程之间的工作负载分配。有时，由于例如网络延迟，资源争用，不可预测的工作量峰值，不可避免地会出现不同步的处理速度。为了避免在这些情况下超时，请确保在调用[init_process_group](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/distributed.html%23torch.distributed.init_process_group)时传递足够大`timeout`value


#### 保存和加载 Checkpoints


在训练期间，通常使用torch.save 和torch.load 来保存和加载 Checkpoints，有关更多详细信息，请参见 [SAVING AND LOADING MODELS](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/beginner/saving_loading_models.html)，使用DDP时，一种优化方法是仅在一个进程中保存模型，然后将其加载到所有进程中，从而减少写开销，这是正确的，因为所有过程都从相同的参数开始，并且梯度在反向传播中同步，因此优化程序应将参数设置为相同的值。如果使用此优化，请确保在保存完成之前不要启动所有进程。此外，在加载模块时，您需要提供适当的 **map_location** 参数，以防止进程进入其他人的设备。如果缺少map_location，torch.load 将首先将模块加载到CPU，然后将每个参数复制到保存位置，这将导致同一台机器上的所有进程使用相同的设备集


```python
def demo_checkpoint(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    rank0_devices = [x - rank * len(device_ids) for x in device_ids]
    device_pairs = zip(rank0_devices, device_ids)
    map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```


#### DDP 和 Model Parallelism 一起使用


DDP还可以与 Model Parallelism一起使用，但是不支持进程内的复制。您需要为每个module 副本创建一个进程，与每个进程的多个副本相比，通常可以提高性能。 这种训练方式在具有巨大的数据量较大的模型时特别有用。使用此功能时，需要小心地实现 multi-GPU model，以避免使用硬编码的设备，因为会将不同的模型副本放置到不同的设备上


```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```


将multi-GPU model 传递给DDP时，不得设置device_ids和output_device，输入和输出数据将通过应用程序或模型forward() 方法放置在适当的设备中。


```python
def demo_model_parallel(rank, world_size):
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    run_demo(demo_basic, 2)
    run_demo(demo_checkpoint, 2)

    if torch.cuda.device_count() >= 8:
        run_demo(demo_model_parallel, 4)
```


### Pytorch中的Distributed Data Parallel与混合精度训练（Apex）


参考文章：


- [Pytorch中的Distributed Data Parallel与混合精度训练（Apex）](https://zhuanlan.zhihu.com/p/105755472)



## BatchNormalization 的多卡同步实现


参考：


- [这么骚！Batch Normalization 还能多卡同步？（附源码解析）](https://www.jianshu.com/p/794f142e9d1a)
- [PyTorch 并行训练指南：单机多卡并行、混合精度、同步 BN 训练](https://bbs.cvmart.net/topics/2672)
- [PyTorch 源码解读之 BN & SyncBN：BN 与 多卡同步 BN 详解](https://zhuanlan.zhihu.com/p/337732517)



### 1. 为何在多卡训练的情况下需要对BN进行同步？


在分类和目标检测任务中，通常 batch_size 会比较大，训练的时候一般是不需要对 BN 进行多卡同步，因为这样的操作会因为 GPU 之间的通信而导致训练速度减慢；


但是对于语义分割等稠密估计问题，分辨率高通常会得到更好的效果，这就需要消耗更多的 GPU 显存，**所以通常 batch 会设置比较小，所以每张卡计算的统计量可能和整体数据样本有较大差异**，那么就有必要实现多卡同步了；


另外，如果使用了 pytorch 的 `torch.nn.DataParallel` 机制，数据被可使用的 GPU 卡分割，每张卡上 BN 层的 batch_size 实际上是为 ![](https://g.yuque.com/gr/latex?%5Cfrac%7BBatchSize%7D%7BnGPU%7D#card=math&code=%5Cfrac%7BBatchSize%7D%7BnGPU%7D&id=b90as)。


### 2. 什么是同步的 BN，具体同步的东西


同步的 BN，主要是每张卡对应的 BN 层，分别计算均值和方差，然后基于每张卡的这两个统计量，计算出统一的均值和方差，然后相互进行同步，这样大家都用相同的均值和方差对输入数据进行归一化的操作；


### 3. 如何实现多卡同步的 BN


BN 的性能和 batch size 有很大的关系。batch size 越大，BN 的统计量也会越准。


然而像检测这样的任务，占用显存较高，一张显卡往往只能拿较少的图片（比如 2 张）来训练，这就导致 BN 的表现变差。一个解决方式是 SyncBN：所有卡共享同一个 BN，得到全局的统计量。


PyTorch 的 SyncBN 分别在 `torch/nn/modules/batchnorm.py` 和 `torch/nn/modules/_functions.py` 做了实现。前者主要负责检查输入合法性，以及根据`momentum`等设置进行传参，调用后者。后者负责计算单卡统计量以及进程间通信。


```python
class SyncBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None):
        super(SyncBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.process_group = process_group
        # gpu_size is set through DistributedDataParallel initialization. This is to ensure that SyncBatchNorm is used
        # under supported condition (single GPU per process)
        self.ddp_gpu_size = None

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'
                             .format(input.dim()))

    def _specify_ddp_gpu_num(self, gpu_size):
        if gpu_size > 1:
            raise ValueError('SyncBatchNorm is only supported for DDP with single GPU per process')
        self.ddp_gpu_size = gpu_size

    def forward(self, input):
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        # 接下来这部分与普通BN差别不大
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # 如果在train模式下，或者关闭track_running_stats，就需要同步全局的均值和方差
        need_sync = self.training or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # 如果不需要同步，SyncBN的行为就与普通BN一致
        if not need_sync:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            if not self.ddp_gpu_size:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            return sync_batch_norm.apply(
                input, self.weight, self.bias, self.running_mean, self.running_var,
                self.eps, exponential_average_factor, process_group, world_size)

    # 把普通BN转为SyncBN, 主要做一些参数拷贝
    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = torch.nn.SyncBatchNorm(module.num_features,
                                                   module.eps, module.momentum,
                                                   module.affine,
                                                   module.track_running_stats,
                                                   process_group)
            if module.affine:
                with torch.no_grad():
                    module_output.weight.copy_(module.weight)
                    module_output.bias.copy_(module.bias)
                # keep requires_grad unchanged
                module_output.weight.requires_grad = module.weight.requires_grad
                module_output.bias.requires_grad = module.bias.requires_grad
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output
```


#### 3.1 forward


单卡上的 BN 会计算该卡对应输入的均值、方差，然后做 Normalize；SyncBN 则需要得到全局的统计量，也就是“所有卡上的输入”对应的均值、方差。一个简单的想法是分两个步骤：


1. 每张卡单独计算其均值，然后做一次同步，得到全局均值
1. 用全局均值去算每张卡对应的方差，然后做一次同步，得到全局方差



但两次同步会消耗更多时间，事实上一次同步就可以实现均值和方差的计算：


![](https://g.yuque.com/gr/latex?%5Csigma%5E2%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em(x_i-%5Cmu)%5E2%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em(x_i%5E2%2B%5Cmu%5E2-2x_i%5Cmu)%5E2%5C%5C%0A%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Emx_i%5E2-%5Cmu%5E2%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Emx_i%5E2-(%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Emx_i)%5E2%0A#card=math&code=%5Csigma%5E2%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%28x_i-%5Cmu%29%5E2%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%28x_i%5E2%2B%5Cmu%5E2-2x_i%5Cmu%29%5E2%5C%5C%0A%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Emx_i%5E2-%5Cmu%5E2%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Emx_i%5E2-%28%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Emx_i%29%5E2%0A&id=Glnwa)


其中 m 是![](https://g.yuque.com/gr/latex?%5Cfrac%7BBatchSize%7D%7BnGPU%7D#card=math&code=%5Cfrac%7BBatchSize%7D%7BnGPU%7D&id=pCOlG)，根据上述公式，需要计算的其实就是 ![](https://g.yuque.com/gr/latex?%5Csum_%7Bi%3D1%7D%5Emx_i#card=math&code=%5Csum_%7Bi%3D1%7D%5Emx_i&id=FhaOP) 和 ![](https://g.yuque.com/gr/latex?%5Csum_%7Bi%3D1%7D%5Emx_i%5E2#card=math&code=%5Csum_%7Bi%3D1%7D%5Emx_i%5E2&id=if1z2)，那么其实每张卡分别计算这两个数值，然后同步求和，即可得到全局的方差，而均值也是相同的操作，这样只需要 1 次，即可完成全局的方差和均值的计算。


![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/sync_bn2.jpeg#crop=0&crop=0&crop=1&crop=1&id=ri1oM&originHeight=392&originWidth=720&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)


实现时，`batchnorm.SyncBatchNorm` 根据自身的超参设置、train/eval 等设置参数，并调用`_functions.SyncBatchNorm`，接口是`def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):` 首先算一下单卡上的均值和方差：


```python
# 这里直接算invstd，也就是 1/(sqrt(var+eps))
mean, invstd = torch.batch_norm_stats(input, eps)
```


然后同步各卡的数据，得到`mean_all`和`invstd_all`，再算出全局的统计量，更新`running_mean`，`running_var`:


```python
# 计算全局的mean和invstd
mean, invstd = torch.batch_norm_gather_stats_with_counts(
    input,
    mean_all,
    invstd_all,
    running_mean,
    running_var,
    momentum,
    eps,
    count_all.view(-1).long().tolist()
)
```


#### 3.2 backward


由于不同的进程共享同一组 BN 参数，因此在 backward 到 BN 前、后都需要做进程的通信，在`_functions.SyncBatchNorm`中实现：


```python
# calculate local stats as well as grad_weight / grad_bias
sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
    grad_output,
    saved_input,
    mean,
    invstd,
    weight,
    self.needs_input_grad[0],
    self.needs_input_grad[1],
    self.needs_input_grad[2]
)
```


算出 weight、bias 的梯度以及 dy ， ![](https://g.yuque.com/gr/latex?%5Cfrac%7Bdy%7D%7Bd%5Cmu%7D#card=math&code=%5Cfrac%7Bdy%7D%7Bd%5Cmu%7D&id=cQqPf)  用于计算 x 的梯度：


```python
# all_reduce 计算梯度之和
sum_dy_all_reduce = torch.distributed.all_reduce(
    sum_dy, torch.distributed.ReduceOp.SUM, process_group, async_op=True)
sum_dy_xmu_all_reduce = torch.distributed.all_reduce(
    sum_dy_xmu, torch.distributed.ReduceOp.SUM, process_group, async_op=True)
# ...
# 根据总的size，对梯度做平均
divisor = count_tensor.sum()
mean_dy = sum_dy / divisor
mean_dy_xmu = sum_dy_xmu / divisor
# backward pass for gradient calculation
grad_input = torch.batch_norm_backward_elemt(
    grad_output,
    saved_input,
    mean,
    invstd,
    weight,
    mean_dy,
    mean_dy_xmu
)
```

---

# 技巧


## 1. 并行训练指南


参考：


- [PyTorch 并行训练指南：单机多卡并行、混合精度、同步 BN 训练](https://bbs.cvmart.net/topics/2672)



### 1.1 为什么不使用 nn.DataParallel


PyTorch 中最常用的并行方式，就是`nn.DataParallel` ，它可以帮助我们通过单进程，将模型和数据加载到多个 GPU 中，控制数据在多个 GPU 之间的流动，协同不同 GPU 上的模型进行并行训练；


使用方式也很简单，代码就只需要一行，如下所示：


`model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])`


其中 `device_ids` 用于指定使用的 GPU，`output_device` 指定汇总地图的 GPU 是哪个。


训练模型如下所示：


```python
# main.py
import torch
import torch.distributed as dist

gpus = [0, 1, 2, 3]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

train_dataset = ...

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=...)

model = ...
model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```


但是为什么不使用它来实现呢？原因是这样的：


在每个训练批次（batch）中，因为模型的权重都是在 **一个进程**上先算出来 然后再把他们分发到每个GPU上，所以**网络通信就成为了一个瓶颈，而GPU使用率也通常很低**。


### 1.2 多进程分布式


#### 1.2.1 torch.distributed


在 1.0 之后，官方终于对分布式的常用方法进行了封装，支持 all-reduce，broadcast，send 和 receive 等等。通过 MPI 实现 CPU 通信，通过 NCCL 实现 GPU 通信。官方也曾经提到用 `DistributedDataParallel` 解决 DataParallel 速度慢，GPU 负载不均衡的问题，目前已经很成熟了。


与 DataParallel 的单进程控制多 GPU 不同，在 distributed 的帮助下，我们只需要编写一份代码，torch 就会自动将其分配给n个进程，分别在n个 GPU 上运行。


和单进程训练不同的是，多进程训练需要注意以下事项：


- 在喂数据的时候，一个batch被分到了好几个进程，每个进程在取数据的时候要确保拿到的是不同的数据（`DistributedSampler`）；
- 要告诉每个进程自己是谁，使用哪块GPU（`args.local_rank`）；
- 在做BatchNormalization的时候要注意同步数据。



#### 1.2.2 使用方式


##### 启动方式的改变


在多进程的启动方面，我们不用自己手写 multiprocess 进行一系列复杂的CPU、GPU分配任务，PyTorch为我们提供了一个很方便的启动器 `torch.distributed.lunch` 用于启动文件，所以我们运行训练代码的方式就变成了这样：


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python \-m torch.distributed.launch \--nproc_per_node=4 main.py
```


其中的 `--nproc_per_node` 参数用于指定为当前主机创建的进程数，由于我们是单机多卡，所以这里node数量为1，所以我们这里设置为所使用的GPU数量即可。


##### 初始化


在启动器为我们启动python脚本后，会通过参数 `local_rank` 来告诉我们当前进程使用的是哪个GPU，用于我们在每个进程中指定不同的device：


```python
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    device = torch.device(f'cuda:{args.local_rank}')
    ...
```


其中 `torch.distributed.init_process_group` 用于初始化GPU通信方式（NCCL）和参数的获取方式（env代表通过环境变量）


##### DataLoader


在读取数据的时候，我们要保证一个batch里的数据被均摊到每个进程上，每个进程都能获取到不同的数据，但如果我们手动去告诉每个进程拿哪些数据的话太麻烦了，PyTorch也为我们封装好了这一方法。


所以我们在初始化 `data loader` 的时候需要使用到 `torch.utils.data.distributed.DistributedSampler` 这个特性：


```python
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
```


这样就能给每个进程一个不同的 sampler，告诉每个进程自己分别取哪些数据。


##### 模型的初始化


和 `nn.DataParallel` 的方式一样，我们对于模型的初始化也是简单的一句话就行了


```python
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
```


### 1.3 NVIDIA/apex 混合精度训练、并行训练、同步BN


#### 1.3.1 介绍


注：需要使用到`Volta结构`的GPU，目前只有Tesla V100和TITAN V系列支持。


Apex 是 NVIDIA 开源的用于混合精度训练和分布式训练库。Apex 对混合精度训练的过程进行了封装，改两三行配置就可以进行混合精度的训练，从而大幅度降低显存占用，节约运算时间。此外，Apex 也提供了对分布式训练的封装，针对 NVIDIA 的 NCCL 通信库进行了优化。


什么是混合精度训练？


[https://zhuanlan.zhihu.com/p/79887894](https://zhuanlan.zhihu.com/p/79887894)


混合精度训练是在尽可能减少精度损失的情况下利用半精度浮点数加速训练。它使用FP16即半精度浮点数存储权重和梯度。在减少占用内存的同时起到了加速训练的效果。


float16和float相比恰里，总结下来就是两个原因：**内存占用更少，计算更快**。


- **内存占用更少**：这个是显然可见的，通用的模型 fp16 占用的内存只需原来的一半。memory-bandwidth 减半所带来的好处： 
   - 模型占用的内存更小，训练的时候可以用更大的batchsize。
   - 模型训练时，通信量（特别是多卡，或者多机多卡）大幅减少，大幅减少等待时间，加快数据的流通。

 

- **计算更快**： 
   - 目前的不少GPU都有针对 fp16 的计算进行优化。论文指出：在近期的GPU中，半精度的计算吞吐量可以是单精度的 2-8 倍；从下图我们可以看到混合精度训练几乎没有性能损失。

​

![apex_fig1.png](https://cdn.nlark.com/yuque/0/2021/png/308996/1621060556235-cf8ab5a3-7339-4968-9e1a-e1a3c3360b85.png#clientId=uf4d31bb9-dd4d-4&crop=0&crop=0&crop=1&crop=1&from=ui&id=u1f3ea558&margin=%5Bobject%20Object%5D&name=apex_fig1.png&originHeight=234&originWidth=720&originalType=binary&ratio=1&rotation=0&showTitle=false&size=48566&status=done&style=none&taskId=u2fd123ab-b5ed-4c49-80e8-0bae29b498e&title=)


#### 1.3.2 使用方式


##### 混合精度


在混合精度训练上，Apex 的封装十分优雅。直接使用 `amp.initialize` 包装模型和优化器，apex 就会自动帮助我们管理模型参数和优化器的精度了，根据精度需求不同可以传入其他配置参数。


```python
from apex import amp

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
```


其中 opt_level 为精度的优化设置，O0（第一个字母是大写字母O）：


- O0：纯FP32训练，可以作为accuracy的baseline；
- O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
- O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算。
- O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；



##### 并行训练


Apex也实现了并行训练模型的转换方式，改动并不大，主要是优化了NCCL的通信，因此代码和 torch.distributed 保持一致，换一下调用的API即可：


```python
from apex import amp
from apex.parallel import DistributedDataParallel

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model = DistributedDataParallel(model, delay_allreduce=True)

# 反向传播时需要调用 amp.scale_loss，用于根据loss值自动对精度进行缩放
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```


##### 同步 BN


Apex为我们实现了同步BN，用于解决单GPU的minibatch太小导致BN在训练时不收敛的问题。


```python
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel

# 注意顺序：三个顺序不能错
model = convert_syncbn_model(UNet3d(n_channels=1, n_classes=1)).to(device)
model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
model = DistributedDataParallel(model, delay_allreduce=True)
```


调用该函数后，Apex会自动遍历model的所有层，将BatchNorm层替换掉。


#### 1.3.3 汇总


Apex的并行训练部分主要与如下代码段有关：


```python
# main.py
import torch
import argparse
import torch.distributed as dist

from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
model = convert_syncbn_model(model)
model, optimizer = amp.initialize(model, optimizer)
model = DistributedDataParallel(model, device_ids=[args.local_rank])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      optimizer.zero_grad()
      with amp.scale_loss(loss, optimizer) as scaled_loss:
         scaled_loss.backward()
      optimizer.step()
```


使用 launch 启动：


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python \-m torch.distributed.launch \--nproc_per_node=4 main.py
```


### 1.4 多卡后的 batch_size 和 learning_rate 的调整


参考：


- [如何理解深度学习分布式训练中的large batch size与learning rate的关系？](https://www.zhihu.com/question/64134994)



从理论上来说，`lr = batch_size * base lr`，因为 batch_size 的增大会导致你 update 次数的减少，所以为了达到相同的效果，应该是同比例增大的。


但是更大的 lr 可能会导致收敛的不够好，尤其是在刚开始的时候，如果你使用很大的 lr，可能会直接爆炸，所以可能会需要一些 warmup 来逐步的把 lr 提高到你想设定的 lr，可以参考论文 [Training ImageNet in 1 Hour](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.02677v1)。


实际应用中发现不一定要同比例增长，有时候可能增大到 `batch_size/2` 倍的效果已经很不错了。


## 2. PyTorch提速


参考：


- [简单两步加速PyTorch里的Dataloader](https://zhuanlan.zhihu.com/p/68191407)
- [如何给你PyTorch里的Dataloader打鸡血](https://zhuanlan.zhihu.com/p/66145913)
- [pytorch dataloader数据加载占用了大部分时间，各位大佬都是怎么解决的？](https://www.zhihu.com/question/307282137/answer/907835663)
- [https://github.com/lartpang/PyTorchTricks](https://github.com/lartpang/PyTorchTricks)
- [给训练踩踩油门 —— Pytorch 加速数据读取](https://zhuanlan.zhihu.com/p/80695364)



PyTorch 提速，包括 dataloader 的提速；


### 预处理提速


- 尽量减少每次读取数据时的预处理操作, 可以考虑把一些固定的操作, 例如 `resize` , 事先处理好保存下来, 训练的时候直接拿来用
- Linux上将预处理搬到GPU上加速: 
   - `NVIDIA/DALI` :[https://github.com/NVIDIA/DALI](https://github.com/NVIDIA/DALI)

 
### IO提速


- 推荐大家关注下mmcv，其对数据的读取提供了比较高效且全面的支持： 
   - OpenMMLab：MMCV 核心组件分析(三): FileClient [https://zhuanlan.zhihu.com/p/339190576](https://zhuanlan.zhihu.com/p/339190576)

 
#### 使用更快的图片处理


- `opencv` 一般要比 `PIL` 要快 （**但是要注意，**`**PIL**`**的惰性加载的策略使得其看上去**`**open**`**要比**`**opencv**`**的**`**imread**`**要快，但是实际上那并没有完全加载数据，可以对**`**open**`**返回的对象调用其**`**load()**`**方法，从而手动加载数据，这时的速度才是合理的**）
- 对于 `jpeg` 读取, 可以尝试 `jpeg4py`
- 存 `bmp` 图(降低解码时间)
- 关于不同图像处理库速度的讨论建议关注下这个：Python的各种imread函数在实现方式和读取速度上有何区别？ - 知乎 [https://www.zhihu.com/question/48762352](https://www.zhihu.com/question/48762352)



#### 小图拼起来存放(降低读取次数)


对于大规模的小文件读取, 建议转成单独的文件, 可以选择的格式可以考虑: `TFRecord（Tensorflow）` , `recordIO（recordIO）` , `hdf5` , `pth` , `n5` , `lmdb` 等等([https://github.com/Lyken17/Efficient-PyTorch#data-loader](https://github.com/Lyken17/Efficient-PyTorch#data-loader))


- `TFRecord` :[https://github.com/vahidk/tfrecord](https://github.com/vahidk/tfrecord)
- 借助 `lmdb` 数据库格式: 
   - [https://github.com/Fangyh09/Image2LMDB](https://github.com/Fangyh09/Image2LMDB)
   - [https://blog.csdn.net/P_LarT/article/details/103208405](https://blog.csdn.net/P_LarT/article/details/103208405)
   - [https://github.com/lartpang/PySODToolBox/blob/master/ForBigDataset/ImageFolder2LMDB.py](https://github.com/lartpang/PySODToolBox/blob/master/ForBigDataset/ImageFolder2LMDB.py)

 
#### 预读取数据


- 预读取下一次迭代需要的数据
- 设置 **num_worker: **DataLoader  的 num_worker 如果设置太小，则起不到多线程提速的作用，如果设置太大，则会造成线程阻塞或者爆内存，导致训练变慢或者程序崩溃，**可以考虑数量是 cpu 的核心数或者 gpu 的数量比较合适**。

​

【参考】


- 如何给你PyTorch里的Dataloader打鸡血 - MKFMIKU的文章 - 知乎 [https://zhuanlan.zhihu.com/p/66145913](https://zhuanlan.zhihu.com/p/66145913)
- 给pytorch 读取数据加速 - 体hi的文章 - 知乎 [https://zhuanlan.zhihu.com/p/72956595](https://zhuanlan.zhihu.com/p/72956595)



##### prefetch_generator
使用 [prefetch_generator](https://link.zhihu.com/?target=https%3A//pypi.org/project/prefetch_generator/) 库在后台加载下一 batch 的数据。
**​**

**安装：**
```python
pip install prefetch_generator
```
**​**

**使用：**
```python
# 新建DataLoaderX类
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
```
然后用 DataLoaderX 替换原本的 DataLoader。
**提速原因：**
> 原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
使用 prefetch_generator，我们可以**保证线程不会等待，每个线程都总有至少一个数据在加载**。


**data_prefetcher**
使用 [data_prefetcher](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py%23L256) 新开 cuda stream 来拷贝 tensor 到 gpu。
**使用：**
```python
class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
```
然后训练代码改写为：
```python
# ----改造前----
for iter_id, batch in enumerate(data_loader):
    if iter_id >= num_iters:
        break
    for k in batch:
        if k != 'meta':
            batch[k] = batch[k].to(device=opt.device, non_blocking=True)
    run_step()
    
# ----改造后----
prefetcher = DataPrefetcher(data_loader, opt)
batch = prefetcher.next()
iter_id = 0
while batch is not None:
    iter_id += 1
    if iter_id >= num_iters:
        break
    run_step()
    batch = prefetcher.next()
```
**提速原因：**
默认情况下，PyTorch 将所有涉及到 GPU 的操作（比如内核操作，cpu->gpu，gpu->cpu）都排入同一个 stream（default stream）中，并对同一个流的操作序列化，它们永远不会并行。要想并行，两个操作必须位于不同的 stream 中。
而前向传播位于 default stream 中，因此，要想将下一个 batch 数据的预读取（涉及 cpu->gpu）与当前 batch 的前向传播并行处理，就必须：
（1） cpu 上的数据 batch 必须 pinned;
（2）预读取操作必须在另一个 stream 上进行
上面的 data_prefetcher 类满足这两个要求。注意 dataloader 必须设置 pin_memory=True 来满足第一个条件。

#### 借助内存


- 直接载到内存里面, 或者把把内存映射成磁盘好了



【参考】


- 参见 [https://zhuanlan.zhihu.com/p/66145913](https://zhuanlan.zhihu.com/p/66145913) 的评论中 @雨宫夏一 的评论（额，似乎那条评论找不到了，不过大家可以搜一搜）



#### 借助固态


- 把读取速度慢的机械硬盘换成 NVME 固态吧～



【参考】


- 如何给你PyTorch里的Dataloader打鸡血 - MKFMIKU的文章 - 知乎 [https://zhuanlan.zhihu.com/p/66145913](https://zhuanlan.zhihu.com/p/66145913)



### 训练策略


**_低精度训练_**


- 在训练中使用低精度(`FP16`甚至`INT8`、二值网络、三值网络)表示取代原有精度(`FP32`)表示 
   - 使用 Apex 的混合精度或者是PyTorch1.6开始提供的torch.cuda.amp模块来训练. 可以节约一定的显存并提速, 但是要小心一些不安全的操作如mean和sum: 
      - `NVIDIA/Apex`: 
         - [https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729)
         - [https://github.com/nvidia/apex](https://github.com/nvidia/apex)
         - Pytorch 安装 APEX 疑难杂症解决方案 - 陈瀚可的文章 - 知乎https://zhuanlan.zhihu.com/p/80386137
         - [http://kevinlt.top/2018/09/14/mixed_precision_training/](http://kevinlt.top/2018/09/14/mixed_precision_training/)

 

      - `torch.cuda.amp`: 
         - PyTorch的文档：[https://pytorch.org/docs/stable/notes/amp_examples.html>](https://pytorch.org/docs/stable/notes/amp_examples.html%3E)

 
### 代码层面


- `torch.backends.cudnn.benchmark = True`
- Do numpy-like operations on the GPU wherever you can
- Free up memory using `del`
- Avoid unnecessary transfer of data from the GPU
- Use pinned memory, and use `non_blocking=True`to parallelize data transfer and GPU number crunching 
   - 文档：[https://pytorch.org/docs/stable/nn.html#torch.nn.Module.to](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.to)
   - 关于 `non_blocking=True` 的设定的一些介绍：Pytorch有什么节省显存的小技巧？ - 陈瀚可的回答 - 知乎 [https://www.zhihu.com/question/274635237/answer/756144739](https://www.zhihu.com/question/274635237/answer/756144739)

 

- 网络设计很重要, 外加不要初始化任何用不到的变量, 因为 PyTorch 的初始化和 `forward` 是分开的, 他不会因为你不去使用, 而不去初始化
- 合适的 `num_worker` : Pytorch 提速指南 - 云梦的文章 - 知乎 https://zhuanlan.zhihu.com/p/39752167(这里也包含了一些其他细节上的讨论)



### 模型设计


来自 ShuffleNetV2 的结论:(内存访问消耗时间, `memory access cost` 缩写为 `MAC` )


- 卷积层输入输出通道一致: 卷积层的输入和输出特征通道数相等时 MAC 最小, 此时模型速度最快
- 减少卷积分组: 过多的 group 操作会增大 MAC, 从而使模型速度变慢
- 减少模型分支: 模型中的分支数量越少, 模型速度越快
- 减少 `element-wise` 操作: `element-wise` 操作所带来的时间消耗远比在 FLOPs 上的体现的数值要多, 因此要尽可能减少 `element-wise` 操作( `depthwise convolution` 也具有低 FLOPs 、高 MAC 的特点)



其他:


- 降低复杂度: 例如模型裁剪和剪枝, 减少模型层数和参数规模
- 改模型结构: 例如模型蒸馏, 通过知识蒸馏方法来获取小模型



### 推理加速


#### 半精度与权重量化


在推理中使用低精度( `FP16` 甚至 `INT8` 、二值网络、三值网络)表示取代原有精度( `FP32` )表示:


- `TensorRT` 是 NVIDIA 提出的神经网络推理(Inference)引擎, 支持训练后 8BIT 量化, 它使用基于交叉熵的模型量化算法, 通过最小化两个分布的差异程度来实现
- Pytorch1.3 开始已经支持量化功能, 基于 QNNPACK 实现, 支持训练后量化, 动态量化和量化感知训练等技术
- 另外 `Distiller` 是 Intel 基于 Pytorch 开源的模型优化工具, 自然也支持 Pytorch 中的量化技术
- 微软的 `NNI` 集成了多种量化感知的训练算法, 并支持 `PyTorch/TensorFlow/MXNet/Caffe2` 等多个开源框架



【参考】:


- 有三AI:【杂谈】当前模型量化有哪些可用的开源工具?[https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649037243&idx=1&sn=db2dc420c4d086fc99c7d8aada767484&chksm=8712a7c6b0652ed020872a97ea426aca1b06adf7571af3da6dac8ce991fd61001245e9bf6e9b&mpshare=1&scene=1&srcid=&sharer_sharetime=1576667804820&sharer_shareid=1d0dbdb37c6b95413d1d4fe7d61ed8f1&exportkey=A6g%2Fj50pMJYVXsedNyDVh9k%3D&pass_ticket=winxjBrzw0kHErbSri5yXS88yBx1a%2BAL9KKTG6Zt1MMS%2FeI2hpx%2BmeaLsrahnlOS#rd](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649037243&idx=1&sn=db2dc420c4d086fc99c7d8aada767484&chksm=8712a7c6b0652ed020872a97ea426aca1b06adf7571af3da6dac8ce991fd61001245e9bf6e9b&mpshare=1&scene=1&srcid=&sharer_sharetime=1576667804820&sharer_shareid=1d0dbdb37c6b95413d1d4fe7d61ed8f1&exportkey=A6g%2Fj50pMJYVXsedNyDVh9k%3D&pass_ticket=winxjBrzw0kHErbSri5yXS88yBx1a%2BAL9KKTG6Zt1MMS%2FeI2hpx%2BmeaLsrahnlOS#rd)



#### 网络 inference 阶段 Conv 层和 BN 层融合


【参考】


- [https://zhuanlan.zhihu.com/p/110552861](https://zhuanlan.zhihu.com/p/110552861)
- PyTorch本身提供了类似的功能, 但是我没有使用过, 希望有朋友可以提供一些使用体会:[https://pytorch.org/docs/1.3.0/quantization.html#torch.quantization.fuse_modules](https://pytorch.org/docs/1.3.0/quantization.html#torch.quantization.fuse_modules)
- 网络inference阶段conv层和BN层的融合 - autocyz的文章 - 知乎 [https://zhuanlan.zhihu.com/p/48005099](https://zhuanlan.zhihu.com/p/48005099)



#### 多分支结构融合成单分支


- ACNet、RepVGG这种设计策略也很有意思 
   - RepVGG|让你的ConVNet一卷到底，plain网络首次超过80%top1精度：[https://mp.weixin.qq.com/s/M4Kspm6hO3W8fXT_JqoEhA](https://mp.weixin.qq.com/s/M4Kspm6hO3W8fXT_JqoEhA)

 
### 时间分析


- Python 的 `cProfile` 可以用来分析.(Python 自带了几个性能分析的模块: `profile` , `cProfile` 和 `hotshot` , 使用方法基本都差不多, 无非模块是纯 Python 还是用 C 写的)



### 项目推荐


- 基于 Pytorch 实现模型压缩([https://github.com/666DZY666/model-compression](https://github.com/666DZY666/model-compression)): 
   - 量化:8/4/2 bits(dorefa)、三值/二值(twn/bnn/xnor-net)
   - 剪枝: 正常、规整、针对分组卷积结构的通道剪枝
   - 分组卷积结构
   - 针对特征二值量化的BN融合

 
### 扩展阅读


- pytorch dataloader数据加载占用了大部分时间, 各位大佬都是怎么解决的? - 知乎 [https://www.zhihu.com/question/307282137](https://www.zhihu.com/question/307282137)
- 使用pytorch时, 训练集数据太多达到上千万张, Dataloader加载很慢怎么办? - 知乎 [https://www.zhihu.com/question/356829360](https://www.zhihu.com/question/356829360)
- PyTorch 有哪些坑/bug? - 知乎 [https://www.zhihu.com/question/67209417](https://www.zhihu.com/question/67209417)
- [https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/](https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/)
- 26秒单GPU训练CIFAR10, Jeff Dean也点赞的深度学习优化技巧 - 机器之心的文章 - 知乎 [https://zhuanlan.zhihu.com/p/79020733](https://zhuanlan.zhihu.com/p/79020733)
- 线上模型加入几个新特征训练后上线, tensorflow serving预测时间为什么比原来慢20多倍? - TzeSing的回答 - 知乎 [https://www.zhihu.com/question/354086469/answer/894235805](https://www.zhihu.com/question/354086469/answer/894235805)
- 相关资料 · 语雀 [https://www.yuque.com/lart/gw5mta/bl3p3y](https://www.yuque.com/lart/gw5mta/bl3p3y)
- ShuffleNetV2:[https://arxiv.org/pdf/1807.11164.pdf](https://arxiv.org/pdf/1807.11164.pdf)
- 今天, 你的模型加速了吗? 这里有5个方法供你参考(附代码解析): [https://mp.weixin.qq.com/s?__biz=MzI0ODcxODk5OA==&mid=2247511633&idx=2&sn=a5ab187c03dfeab4e64c85fc562d7c0d&chksm=e99e9da8dee914be3d713c41d5dedb7fcdc9982c8b027b5e9b84e31789913c5b2dd880210ead&mpshare=1&scene=1&srcid=&sharer_sharetime=1576934236399&sharer_shareid=1d0dbdb37c6b95413d1d4fe7d61ed8f1&exportkey=A%2B3SqYGse83qyFva%2BYSy3Ng%3D&pass_ticket=winxjBrzw0kHErbSri5yXS88yBx1a%2BAL9KKTG6Zt1MMS%2FeI2hpx%2BmeaLsrahnlOS#rd](https://mp.weixin.qq.com/s?__biz=MzI0ODcxODk5OA==&mid=2247511633&idx=2&sn=a5ab187c03dfeab4e64c85fc562d7c0d&chksm=e99e9da8dee914be3d713c41d5dedb7fcdc9982c8b027b5e9b84e31789913c5b2dd880210ead&mpshare=1&scene=1&srcid=&sharer_sharetime=1576934236399&sharer_shareid=1d0dbdb37c6b95413d1d4fe7d61ed8f1&exportkey=A%2B3SqYGse83qyFva%2BYSy3Ng%3D&pass_ticket=winxjBrzw0kHErbSri5yXS88yBx1a%2BAL9KKTG6Zt1MMS%2FeI2hpx%2BmeaLsrahnlOS#rd)
- pytorch常见的坑汇总 - 郁振波的文章 - 知乎 [https://zhuanlan.zhihu.com/p/77952356](https://zhuanlan.zhihu.com/p/77952356)
- Pytorch 提速指南 - 云梦的文章 - 知乎 [https://zhuanlan.zhihu.com/p/39752167](https://zhuanlan.zhihu.com/p/39752167)



## 3. PyTorch 节省显存


参考：


- [Pytorch有什么节省内存(显存)的小技巧? ](https://www.zhihu.com/question/274635237)



### 尽量使用 `inplace` 操作


尽可能使用 `inplace` 操作, 比如 `relu` 可以使用 `inplace=True` . 一个简单的使用方法, 如下:


```python
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

model.apply(inplace_relu)
```


进一步, 比如ResNet和DenseNet可以将 `batchnorm` 和 `relu` 打包成 `inplace` , 在bp时再重新计算. 使用到了pytorch新的 `checkpoint` 特性, 有以下两个代码. 由于需要重新计算bn后的结果, 所以会慢一些.


- [gpleiss/efficient_densenet_pytorch](https://github.com/gpleiss/efficient_densenet_pytorch)
- [In-Place Activated BatchNorm:mapillary/inplace_abn](https://github.com/mapillary/inplace_abn)



### 删除loss


每次循环结束时删除 loss, 可以节约很少显存, 但聊胜于无. 可见如下issue: [Tensor to Variable and memory freeing best practices](https://discuss.pytorch.org/t/tensor-to-variable-and-memory-freeing-best-practices/6000/2)


### 混合精度


使用 `Apex` 的混合精度或者是PyTorch1.6开始提供的`torch.cuda.amp`模块来训练. 可以节约一定的显存, 但是要小心一些不安全的操作如mean和sum:


- `NVIDIA/Apex`: 
   - [https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729)
   - [https://github.com/nvidia/apex](https://github.com/nvidia/apex)
   - Pytorch 安装 APEX 疑难杂症解决方案 - 陈瀚可的文章 - 知乎https://zhuanlan.zhihu.com/p/80386137
   - [http://kevinlt.top/2018/09/14/mixed_precision_training/](http://kevinlt.top/2018/09/14/mixed_precision_training/)

 

- `torch.cuda.amp`: 
   - PyTorch的文档：[https://pytorch.org/docs/stable/notes/amp_examples.html](https://pytorch.org/docs/stable/notes/amp_examples.html)

 
### 对不需要反向传播的操作进行管理


- 对于不需要bp的forward, 如validation 请使用 `torch.no_grad` , 注意 `model.eval()` 不等于 `torch.no_grad()` , 请看如下讨论: ['model.eval()' vs 'with torch.no_grad()'](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
- ~~将不需要更新的层的参数从优化器中排除~~将变量的 `requires_grad`设为 `False`, 让变量不参与梯度的后向传播(主要是为了减少不必要的梯度的显存占用)



### 显存清理


- `torch.cuda.empty_cache()`：这是`del`的进阶版, 使用`nvidia-smi`会发现显存有明显的变化. 但是训练时最大的显存占用似乎没变. 大家可以试试: [How can we release GPU memory cache?](https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530)
- 可以使用 `del` 删除不必要的中间变量, 或者使用 `replacing variables` 的形式来减少占用.



### 梯度累加


- 把一个 `batchsize=64` 分为两个32的batch, 两次forward以后, backward一次. 但会影响 `batchnorm` 等和 `batchsize` 相关的层. 在PyTorch的文档 https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation中提到了梯度累加与混合精度并用的例子.
- 这篇文章提到使用梯度累加技术实现对于分布式训练的加速：[https://zhuanlan.zhihu.com/p/250471767](https://zhuanlan.zhihu.com/p/250471767)



### 使用 `checkpoint` 技术


#### `torch.utils.checkpoint`


**这是更为通用的选择.**


【参考】


- [https://blog.csdn.net/one_six_mix/article/details/93937091](https://blog.csdn.net/one_six_mix/article/details/93937091)
- [https://pytorch.org/docs/1.3.0/_modules/torch/utils/checkpoint.html#checkpoint](https://pytorch.org/docs/1.3.0/_modules/torch/utils/checkpoint.html#checkpoint)



#### Training Deep Nets with Sublinear Memory Cost


方法来自论文: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174). 训练 CNN 时, Memory 主要的开销来自于储存用于计算 backward 的 activation, 一般的 workflow 是这样的


![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/pytorch_memory_1.gif#crop=0&crop=0&crop=1&crop=1&id=NjYfC&originHeight=121&originWidth=541&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)


对于一个长度为 N 的 CNN, 需要 O(N) 的内存. 这篇论文给出了一个思路, 每隔 sqrt(N) 个 node 存一个 activation, 中需要的时候再算, 这样显存就从 O(N) 降到了 O(sqrt(N)).


![pytorch_memory_2.gif](https://cdn.nlark.com/yuque/0/2021/gif/308996/1621060617712-ba565e08-94ac-4eba-ac53-5d5f8df47adc.gif#clientId=uf4d31bb9-dd4d-4&crop=0&crop=0&crop=1&crop=1&from=ui&id=uc558155c&margin=%5Bobject%20Object%5D&name=pytorch_memory_2.gif&originHeight=121&originWidth=541&originalType=binary&ratio=1&rotation=0&showTitle=false&size=312403&status=done&style=none&taskId=ue209b634-72f7-4d9a-b63b-8e7281f1561&title=)


对于越深的模型, 这个方法省的显存就越多, 且速度不会明显变慢.


![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/pytorch_memory_3.png#crop=0&crop=0&crop=1&crop=1&id=i5wCc&originHeight=376&originWidth=720&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)


实现的版本 [https://github.com/Lyken17/pytorch-memonger](https://github.com/Lyken17/pytorch-memonger)


本文章首发在 极市计算机视觉技术社区


【参考】: Pytorch有什么节省内存(显存)的小技巧? - Lyken的回答 - 知乎 [https://www.zhihu.com/question/274635237/answer/755102181](https://www.zhihu.com/question/274635237/answer/755102181)


### 相关工具


- These codes can help you to detect your GPU memory during training with Pytorch. [https://github.com/Oldpan/Pytorch-Memory-Utils](https://github.com/Oldpan/Pytorch-Memory-Utils)
- Just less than nvidia-smi? [https://github.com/wookayin/gpustat](https://github.com/wookayin/gpustat)



### 参考资料


- Pytorch有什么节省内存(显存)的小技巧? - 郑哲东的回答 - 知乎 [https://www.zhihu.com/question/274635237/answer/573633662](https://www.zhihu.com/question/274635237/answer/573633662)
- 浅谈深度学习: 如何计算模型以及中间变量的显存占用大小 [https://oldpan.me/archives/how-to-calculate-gpu-memory](https://oldpan.me/archives/how-to-calculate-gpu-memory)
- 如何在Pytorch中精细化利用显存 [https://oldpan.me/archives/how-to-use-memory-pytorch](https://oldpan.me/archives/how-to-use-memory-pytorch)
- Pytorch有什么节省显存的小技巧? - 陈瀚可的回答 - 知乎:[https://www.zhihu.com/question/274635237/answer/756144739](https://www.zhihu.com/question/274635237/answer/756144739)



## 4. 其他技巧


### 重现


#### 强制确定性操作


**PyTorch 1.7 新更新功能**


```
>>> import torch
>>> torch.set_deterministic(True)
```


【参考】:[https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms)


#### 设置随机数种子


```python
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()
```


参考:[https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch%E9%9A%8F%E6%9C%BA%E7%A7%8D%E5%AD%90/](https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch%E9%9A%8F%E6%9C%BA%E7%A7%8D%E5%AD%90/)
















[

](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/)


