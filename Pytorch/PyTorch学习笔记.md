# 基础



## 张量数据结构

参考：

- [2-1,张量数据结构](https://github.com/lyhue1991/eat_pytorch_in_20_days/blob/master/2-1,%E5%BC%A0%E9%87%8F%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84.md)



### 1. 张量的数据类型

张量的数据类型和numpy.array基本一一对应，但是不支持str类型。

包括:

```python
torch.float64(torch.double),
torch.float32(torch.float),
torch.float16,
torch.int64(torch.long),
torch.int32(torch.int),
torch.int16,
torch.int8,
torch.uint8,
torch.bool
```



一般神经网络建模使用的都是torch.float32类型。

```python
import numpy as np
import torch 

# 自动推断数据类型

i = torch.tensor(1);print(i,i.dtype)
x = torch.tensor(2.0);print(x,x.dtype)
b = torch.tensor(True);print(b,b.dtype)
# 输出结果
tensor(1) torch.int64
tensor(2.) torch.float32
tensor(True) torch.bool

# 指定数据类型
i = torch.tensor(1,dtype = torch.int32);print(i,i.dtype)
x = torch.tensor(2.0,dtype = torch.double);print(x,x.dtype)
# 输出结果
tensor(1, dtype=torch.int32) torch.int32
tensor(2., dtype=torch.float64) torch.float64

# 使用特定类型构造函数
i = torch.IntTensor(1);print(i,i.dtype)
x = torch.Tensor(np.array(2.0));print(x,x.dtype) #等价于torch.FloatTensor
b = torch.BoolTensor(np.array([1,0,2,0])); print(b,b.dtype)
# 输出结果
tensor([5], dtype=torch.int32) torch.int32
tensor(2.) torch.float32
tensor([ True, False,  True, False]) torch.bool

# 不同类型进行转换
i = torch.tensor(1); print(i,i.dtype)
x = i.float(); print(x,x.dtype) #调用 float方法转换成浮点类型
y = i.type(torch.float); print(y,y.dtype) #使用type函数转换成浮点类型
z = i.type_as(x);print(z,z.dtype) #使用type_as方法转换成某个Tensor相同类型
# 输出结果
tensor(1) torch.int64
tensor(1.) torch.float32
tensor(1.) torch.float32
tensor(1.) torch.float32
```



### 2. 张量的维度

不同类型的数据可以用不同维度(dimension)的张量来表示。

- 标量为0维张量
- 向量为1维张量
- 矩阵为2维张量。
- 彩色图像有rgb三个通道，可以表示为3维张量
- 视频还有时间维，可以表示为4维张量。

可以简单地总结为：有几层中括号，就是多少维的张量。

```python
scalar = torch.tensor(True)
print(scalar)
print(scalar.dim())  # 标量，0维张量
```

输出：

```
tensor(True)
0
```



```python
vector = torch.tensor([1.0,2.0,3.0,4.0]) #向量，1维张量
print(vector)
print(vector.dim())
# 输出结果
tensor([1., 2., 3., 4.])
1

matrix = torch.tensor([[1.0,2.0],[3.0,4.0]]) #矩阵, 2维张量
print(matrix)
print(matrix.dim())
matrix = torch.tensor([[1.0,2.0],[3.0,4.0]]) #矩阵, 2维张量
print(matrix)
print(matrix.dim())

tensor3 = torch.tensor([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])  # 3维张量
print(tensor3)
print(tensor3.dim())
# 输出
tensor([[[1., 2.],
         [3., 4.]],

        [[5., 6.],
         [7., 8.]]])
3

tensor4 = torch.tensor([[[[1.0,1.0],[2.0,2.0]],[[3.0,3.0],[4.0,4.0]]],
                        [[[5.0,5.0],[6.0,6.0]],[[7.0,7.0],[8.0,8.0]]]])  # 4维张量
print(tensor4)
print(tensor4.dim())
# 输出
tensor([[[[1., 1.],
          [2., 2.]],

         [[3., 3.],
          [4., 4.]]],


        [[[5., 5.],
          [6., 6.]],

         [[7., 7.],
          [8., 8.]]]])
4
```



### . 张量的尺寸

查看尺寸的方法：

- 使用 `shape` 属性
-  `size()` 方法

改变尺寸方法：

- 可以使用 `view` 方法改变张量的尺寸。
- 如果 `view` 方法改变尺寸失败，可以使用 `reshape` 方法.

代码示例：

```python
scalar = torch.tensor(True)
print(scalar.size())
print(scalar.shape)

torch.Size([])
torch.Size([4])

vector = torch.tensor([1.0,2.0,3.0,4.0])
print(vector.size())
print(vector.shape)
torch.Size([4])
torch.Size([4])

matrix = torch.tensor([[1.0,2.0],[3.0,4.0]])
print(matrix.size())
torch.Size([2, 2])

# 使用view可以改变张量尺寸
vector = torch.arange(0,12)
print(vector)
print(vector.shape)

matrix34 = vector.view(3,4)
print(matrix34)
print(matrix34.shape)

matrix43 = vector.view(4,-1) #-1表示该位置长度由程序自动推断
print(matrix43)
print(matrix43.shape)
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
torch.Size([12])
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
torch.Size([3, 4])
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])
torch.Size([4, 3])

# 有些操作会让张量存储结构扭曲，直接使用view会失败，可以用reshape方法
matrix26 = torch.arange(0,12).view(2,6)
print(matrix26)
print(matrix26.shape)

# 转置操作让张量存储结构扭曲
matrix62 = matrix26.t()
print(matrix62.is_contiguous())


# 直接使用view方法会失败，可以使用reshape方法
#matrix34 = matrix62.view(3,4) #error!
matrix34 = matrix62.reshape(3,4) #等价于matrix34 = matrix62.contiguous().view(3,4)
print(matrix34)
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
torch.Size([2, 6])
False
tensor([[ 0,  6,  1,  7],
        [ 2,  8,  3,  9],
        [ 4, 10,  5, 11]])
```



### 4. 张量和 numpy 数组

张量和 numpy 的数组是可以相互变换的，可以通过 numpy 方法从张量得到 numpy 数组，也可以通过 `torch.from_numpy` 从 numpy 数组变为张量，这两种方法关联的张量和 numpy 数组是共享内存的，即改变其中一个数值，另一个也会改变，如果想避免这种情况，可以采用张量的 `clone` 方法；

还可以使用 `item` 方法从标量张量得到对应的Python数值。

使用 `tolist` 方法从张量得到对应的Python数值列表。

代码示例：

```python
import numpy as np
import torch 

#torch.from_numpy函数从numpy数组得到Tensor
arr = np.zeros(3)
tensor = torch.from_numpy(arr)
print("before add 1:")
print(arr)
print(tensor)

print("\nafter add 1:")
np.add(arr,1, out = arr) #给 arr增加1，tensor也随之改变
print(arr)
print(tensor)
```

输出结果：

```
before add 1:
[0. 0. 0.]
tensor([0., 0., 0.], dtype=torch.float64)

after add 1:
[1. 1. 1.]
tensor([1., 1., 1.], dtype=torch.float64)
```



```python
# numpy方法从Tensor得到numpy数组
tensor = torch.zeros(3)
arr = tensor.numpy()
print("before add 1:")
print(tensor)
print(arr)

print("\nafter add 1:")

#使用带下划线的方法表示计算结果会返回给调用 张量
tensor.add_(1) #给 tensor增加1，arr也随之改变 
#或： torch.add(tensor,1,out = tensor)
print(tensor)
print(arr)
```

输出结果：

```
before add 1:
tensor([0., 0., 0.])
[0. 0. 0.]

after add 1:
tensor([1., 1., 1.])
[1. 1. 1.]
```



```python
# 可以用clone() 方法拷贝张量，中断这种关联
tensor = torch.zeros(3)

#使用clone方法拷贝张量, 拷贝后的张量和原始张量内存独立
arr = tensor.clone().numpy() # 也可以使用tensor.data.numpy()
print("before add 1:")
print(tensor)
print(arr)

print("\nafter add 1:")

#使用 带下划线的方法表示计算结果会返回给调用 张量
tensor.add_(1) #给 tensor增加1，arr不再随之改变
print(tensor)
print(arr)
```

输出结果：

```
before add 1:
tensor([0., 0., 0.])
[0. 0. 0.]

after add 1:
tensor([1., 1., 1.])
[0. 0. 0.]
```



```python
# item方法和tolist方法可以将张量转换成Python数值和数值列表
scalar = torch.tensor(1.0)
s = scalar.item()
print(s)
print(type(s))

tensor = torch.rand(2,2)
t = tensor.tolist()
print(t)
print(type(t))
# 输出结果
1.0
<class 'float'>
[[0.8211846351623535, 0.20020723342895508], [0.011571824550628662, 0.2906131148338318]]
<class 'list'>
```



## 自动微分机制

参考：

- [2-2,自动微分机制](https://github.com/lyhue1991/eat_pytorch_in_20_days/blob/master/2-2,%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E6%9C%BA%E5%88%B6.md)



Pytorch的自动微分机制：

- Pytorch一般通过反向传播 `backward` 方法实现这种求梯度计算。该方法求得的梯度将存在对应自变量张量的 `grad` 属性下。

- 除此之外，也能够调用 `torch.autograd.grad` 函数来实现求梯度计算。



### 1. 利用 backward 方法求导数

backward 方法通常在一个**标量张量**上调用，该方法求得的梯度将存在对应自变量张量的grad属性下。

如果调用的张量非标量，则要传入一个和它**同形状的 gradient 参数张量**。

相当于用该 gradient 参数张量与调用张量作向量点乘，得到的标量结果再反向传播。

#### 标量的反向传播

```python
import numpy as np 
import torch 

# f(x) = a*x**2 + b*x + c的导数
x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c 

y.backward()
dy_dx = x.grad
print(dy_dx)
```

输出结果：

```
tensor(-2.)
```

#### 非标量的反向传播

```python
import numpy as np 
import torch 

# f(x) = a*x**2 + b*x + c
x = torch.tensor([[0.0,0.0],[1.0,2.0]],requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c 

gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])

print("x:\n",x)
print("y:\n",y)
y.backward(gradient = gradient)
x_grad = x.grad
print("x_grad:\n",x_grad)

```

输出结果：

```python
x:
 tensor([[0., 0.],
        [1., 2.]], requires_grad=True)
y:
 tensor([[1., 1.],
        [0., 1.]], grad_fn=<AddBackward0>)
x_grad:
 tensor([[-2., -2.],
        [ 0.,  2.]])

```

#### 非标量的反向传播可以用标量的反向传播实现

```python
import numpy as np 
import torch 

# f(x) = a*x**2 + b*x + c
x = torch.tensor([[0.0,0.0],[1.0,2.0]],requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c 

gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])
z = torch.sum(y*gradient)

print("x:",x)
print("y:",y)
z.backward()
x_grad = x.grad
print("x_grad:\n",x_grad)

```

输出结果

```
x: tensor([[0., 0.],
        [1., 2.]], requires_grad=True)
y: tensor([[1., 1.],
        [0., 1.]], grad_fn=<AddBackward0>)
x_grad:
 tensor([[-2., -2.],
        [ 0.,  2.]])
```



### 2. 利用 autograd.grad 方法求导数

```python
import numpy as np 
import torch 

# f(x) = a*x**2 + b*x + c的导数
x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
y = a*torch.pow(x,2) + b*x + c

# create_graph 设置为 True 将允许创建更高阶的导数 
dy_dx = torch.autograd.grad(y,x,create_graph=True)[0]
print(dy_dx.data)

# 求二阶导数
dy2_dx2 = torch.autograd.grad(dy_dx,x)[0] 
print(dy2_dx2.data)
```

输出结果：

```
tensor(-2.)
tensor(2.)
```

对多个自变量求导数：

```python
import numpy as np 
import torch 

x1 = torch.tensor(1.0,requires_grad = True) # x需要被求导
x2 = torch.tensor(2.0,requires_grad = True)

y1 = x1*x2
y2 = x1+x2


# 允许同时对多个自变量求导数
(dy1_dx1,dy1_dx2) = torch.autograd.grad(outputs=y1,inputs = [x1,x2],retain_graph = True)
print(dy1_dx1,dy1_dx2)

# 如果有多个因变量，相当于把多个因变量的梯度结果求和
(dy12_dx1,dy12_dx2) = torch.autograd.grad(outputs=[y1,y2],inputs = [x1,x2])
print(dy12_dx1,dy12_dx2)
```

输出结果：

```
tensor(2.) tensor(1.)
tensor(3.) tensor(2.)
```





### 3. 利用自动微分和优化器求最小值

```python
import numpy as np 
import torch 

# f(x) = a*x**2 + b*x + c的最小值
x = torch.tensor(0.0,requires_grad = True) # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x],lr = 0.01)

def f(x):
    result = a*torch.pow(x,2) + b*x + c 
    return(result)

for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
   
    
print("y=",f(x).data,";","x=",x.data)
```

输出结果：

```
y= tnsor(0.) ; x= tensor(1.0000)
```





## 动态计算图

参考：

- [2.3 动态计算图](https://github.com/lyhue1991/eat_pytorch_in_20_days/blob/master/2-3,%E5%8A%A8%E6%80%81%E8%AE%A1%E7%AE%97%E5%9B%BE.md)



### 1. 简介

<img src="images/torch%E5%8A%A8%E6%80%81%E5%9B%BE.gif" style="zoom:80%;" />

Pytorch的计算图由节点和边组成，节点表示张量或者Function，边表示张量和Function之间的依赖关系。

Pytorch中的计算图是动态图。这里的动态主要有两重含义。

第一层含义是：**计算图的正向传播是立即执行的**。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，并立即执行正向传播得到计算结果。

第二层含义是：**计算图在反向传播后立即销毁**。下次调用需要重新构建计算图。如果在程序中使用了 `backward` 方法执行了反向传播，或者利用 `torch.autograd.grad` 方法计算了梯度，那么创建的计算图会被立即销毁，释放存储空间，下次调用需要重新创建。

#### 计算图的正向传播是立即执行的

```python
import torch 
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.randn(10,2)
Y = torch.randn(10,1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y,2))

print(loss.data)
print(Y_hat.data)
# 输出结果
tensor(17.8969)
tensor([[3.2613],
        [4.7322],
        [4.5037],
        [7.5899],
        [7.0973],
        [1.3287],
        [6.1473],
        [1.3492],
        [1.3911],
        [1.2150]])
```

#### 计算图在反向传播后立即销毁

```python
import torch 
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.randn(10,2)
Y = torch.randn(10,1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y_hat-Y,2))

#计算图在反向传播后立即销毁，如果需要保留计算图, 需要设置retain_graph = True
loss.backward()  #loss.backward(retain_graph = True) 

#loss.backward() #如果再次执行反向传播将报错

```



### 2. 计算图中的函数

计算图中另一种节点就是函数，即各种对张量操作的函数，它们和 python 中的函数有一个较大的区别，就是它们是同时包含正向计算逻辑和反向传播的逻辑。

可以通过继承 `torch.autograd.Function` 来创建这种支持反向传播的函数：

```python
class MyReLU(torch.autograd.Function):
   
    #正向传播逻辑，可以用ctx存储一些值，供反向传播使用。
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    #反向传播逻辑
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
```

开始调用这种函数：

```python
import torch 
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.tensor([[-1.0,-1.0],[1.0,1.0]])
Y = torch.tensor([[2.0,3.0]])

relu = MyReLU.apply # relu现在也可以具有正向传播和反向传播功能
Y_hat = relu(X@w.t() + b)
loss = torch.mean(torch.pow(Y_hat-Y,2))

loss.backward()

print(w.grad)
print(b.grad)
# 输出结果
tensor([[4.5000, 4.5000]])
tensor([[4.5000]])
# Y_hat的梯度函数即是我们自己所定义的 MyReLU.backward
print(Y_hat.grad_fn)
<torch.autograd.function.MyReLUBackward object at 0x1205a46c8>
```



### 3. 计算图与反向传播

简单地理解一下反向传播的原理和过程。理解该部分原理需要一些高等数学中求导链式法则的基础知识。

```python
import torch 

x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

loss.backward()
```

`loss.backward()` 语句调用后，依次发生以下计算过程。

1. loss 自己的grad梯度赋值为1，即对自身的梯度为1。
2. loss 根据其自身梯度以及关联的backward方法，计算出其对应的自变量即 y1 和 y2 的梯度，将该值赋值到 y1.grad 和 y2.grad。
3. y2 和 y1根据其自身梯度以及关联的backward方法, 分别计算出其对应的自变量x的梯度，x.grad 将其收到的多个梯度值累加。

（注意，1,2,3 步骤的求梯度顺序和对多个梯度值的累加规则恰好是求导链式法则的程序表述）

正因为求导链式法则衍生的梯度累加规则，张量的grad梯度不会自动清零，在需要的时候需要手动置零。



### 4. 叶子节点和非叶子节点

什么是叶子节点张量呢？叶子节点张量需要满足两个条件。

1. 叶子节点张量是**由用户直接创建的张量**，而非由某个Function通过计算得到的张量。
2. 叶子节点张量的 `requires_grad` 属性必须为True.

Pytorch设计这样的规则主要是为了**节约内存或者显存空间**，因为几乎所有的时候，**用户只会关心他自己直接创建的张量的梯度**。

所有依赖于叶子节点张量的张量, 其 `requires_grad` 属性必定是True的，但其梯度值只在计算过程中被用到，不会最终存储到grad属性中。

**如果需要保留中间计算结果的梯度到grad属性中，可以使用 `retain_grad` 方法**。 如果仅仅是为了调试代码查看梯度值，可以利用`register_hook` 打印日志。



执行下面代码，我们会发现 loss.grad并不是我们期望的1,而是 None。

类似地 y1.grad 以及 y2.grad也是 None.

这是为什么呢？这是由于它们不是叶子节点张量。

在反向传播过程中，只有 is_leaf=True 的叶子节点，需要求导的张量的导数结果才会被最后保留下来。

```python
import torch 

x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

loss.backward()
print("loss.grad:", loss.grad)
print("y1.grad:", y1.grad)
print("y2.grad:", y2.grad)
print(x.grad)

# 输出结果
loss.grad: None
y1.grad: None
y2.grad: None
tensor(4.)
```

调用 `is_leaf` 方法查看当前变量是否叶子节点：

```python
print(x.is_leaf)
print(y1.is_leaf)
print(y2.is_leaf)
print(loss.is_leaf)
True
False
False
False
```

利用 `retain_grad` 可以保留非叶子节点的梯度值，利用 `register_hook` 可以查看非叶子节点的梯度值。

```python
import torch 

#正向传播
x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

#非叶子节点梯度显示控制
y1.register_hook(lambda grad: print('y1 grad: ', grad))
y2.register_hook(lambda grad: print('y2 grad: ', grad))
loss.retain_grad()

#反向传播
loss.backward()
print("loss.grad:", loss.grad)
print("x.grad:", x.grad)
# 输出结果
y2 grad:  tensor(4.)
y1 grad:  tensor(-4.)
loss.grad: tensor(1.)
x.grad: tensor(4.)
```



### 5. 计算图在TensorBoard中的可视化

可以利用`torch.utils.tensorboard` 将计算图导出到 TensorBoard进行可视化。

首先创建一个网络模型：

```python
from torch import nn 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Parameter(torch.randn(2,1))
        self.b = nn.Parameter(torch.zeros(1,1))

    def forward(self, x):
        y = x@self.w + self.b
        return y
net = Net()
```

通过 `torch.utils.tensorboard` 将计算图导出到 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./data/tensorboard')
writer.add_graph(net,input_to_model = torch.rand(10,2))
writer.close()
```

可视化

```python
%load_ext tensorboard
#%tensorboard --logdir ./data/tensorboard
from tensorboard import notebook
notebook.list() 
#在tensorboard中查看模型
notebook.start("--logdir ./data/tensorboard")
```





------

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



------

## PyTorch 中 backward() 详解

参考：

- [PyTorch 的 backward 为什么有一个 grad_variables 参数？](https://zhuanlan.zhihu.com/p/29923090)
- [详解Pytorch 自动微分里的（vector-Jacobian product）](https://zhuanlan.zhihu.com/p/65609544)
- [PyTorch 中 backward() 详解](https://www.pytorchtutorial.com/pytorch-backward/)





------

## PyTorch中的钩子（Hook）

参考：

- [pytorch中的钩子（Hook）有何作用](https://www.zhihu.com/question/61044004)
- [Pytorch中autograd以及hook函数详解](https://oldpan.me/archives/pytorch-autograd-hook)



非常形象的定义：(pytorch中的钩子（Hook）有何作用？ - 马索萌的回答 - 知乎 https://www.zhihu.com/question/61044004/answer/294829738)



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
2. **DataParallel是单进程，多线程，并且只能在单台计算机上运行**，而DistributedDataParallel是**多进程**，并且可以在**单机和分布式训练**中使用。因此，即使在单机训练中，您的数据足够小以适合单机，DistributedDataParallel仍要比DataParallel更快。 
3. DistributedDataParallel还可以**预先复制模型**，而不是在每次迭代时复制模型，并且可以避免PIL全局解释器锁定。
4. 如果数据和模型同时很大而无法用一个GPU训练，则可以将model parallel（与DistributedDataParallel结合使用。在这种情况下，每个DistributedDataParallel进程都可以model parallel，并且所有进程共同用数据并行



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

另外，如果使用了 pytorch 的 `torch.nn.DataParallel` 机制，数据被可使用的 GPU 卡分割，每张卡上 BN 层的 batch_size 实际上是为 $\frac{BatchSize}{nGPU}$。



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
2. 用全局均值去算每张卡对应的方差，然后做一次同步，得到全局方差

但两次同步会消耗更多时间，事实上一次同步就可以实现均值和方差的计算：
$$
\sigma^2=\frac{1}{m}\sum_{i=1}^m(x_i-\mu)^2=\frac{1}{m}\sum_{i=1}^m(x_i^2+\mu^2-2x_i\mu)^2\\
=\frac{1}{m}\sum_{i=1}^mx_i^2-\mu^2=\frac{1}{m}\sum_{i=1}^mx_i^2-(\frac{1}{m}\sum_{i=1}^mx_i)^2
$$
其中 m 是$\frac{BatchSize}{nGPU}$，根据上述公式，需要计算的其实就是 $\sum_{i=1}^mx_i$ 和 $\sum_{i=1}^mx_i^2$，那么其实每张卡分别计算这两个数值，然后同步求和，即可得到全局的方差，而均值也是相同的操作，这样只需要 1 次，即可完成全局的方差和均值的计算。

<img src="https://gitee.com/lcai013/image_cdn/raw/master/notes_images/sync_bn2.jpeg" style="zoom:67%;" />

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

算出 weight、bias 的梯度以及 dy ， $\frac{dy}{d\mu}$  用于计算 x 的梯度：

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





------

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

https://zhuanlan.zhihu.com/p/79887894

混合精度训练是在尽可能减少精度损失的情况下利用半精度浮点数加速训练。它使用FP16即半精度浮点数存储权重和梯度。在减少占用内存的同时起到了加速训练的效果。

float16和float相比恰里，总结下来就是两个原因：**内存占用更少，计算更快**。

- **内存占用更少**：这个是显然可见的，通用的模型 fp16 占用的内存只需原来的一半。memory-bandwidth 减半所带来的好处：
  - 模型占用的内存更小，训练的时候可以用更大的batchsize。
  - 模型训练时，通信量（特别是多卡，或者多机多卡）大幅减少，大幅减少等待时间，加快数据的流通。
- **计算更快**：
  - 目前的不少GPU都有针对 fp16 的计算进行优化。论文指出：在近期的GPU中，半精度的计算吞吐量可以是单精度的 2-8 倍；从下图我们可以看到混合精度训练几乎没有性能损失。

![img](images/apex_fig1.png)

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
- https://github.com/lartpang/PyTorchTricks



PyTorch 提速，包括 dataloader 的提速；



### 预处理提速

- 尽量减少每次读取数据时的预处理操作, 可以考虑把一些固定的操作, 例如 `resize` , 事先处理好保存下来, 训练的时候直接拿来用
- Linux上将预处理搬到GPU上加速:
  - `NVIDIA/DALI` :https://github.com/NVIDIA/DALI

### IO提速

- 推荐大家关注下mmcv，其对数据的读取提供了比较高效且全面的支持：
  - OpenMMLab：MMCV 核心组件分析(三): FileClient https://zhuanlan.zhihu.com/p/339190576

#### 使用更快的图片处理

- `opencv` 一般要比 `PIL` 要快 （**但是要注意，`PIL`的惰性加载的策略使得其看上去`open`要比`opencv`的`imread`要快，但是实际上那并没有完全加载数据，可以对`open`返回的对象调用其`load()`方法，从而手动加载数据，这时的速度才是合理的**）
- 对于 `jpeg` 读取, 可以尝试 `jpeg4py`
- 存 `bmp` 图(降低解码时间)
- 关于不同图像处理库速度的讨论建议关注下这个：Python的各种imread函数在实现方式和读取速度上有何区别？ - 知乎 https://www.zhihu.com/question/48762352



#### 小图拼起来存放(降低读取次数)

对于大规模的小文件读取, 建议转成单独的文件, 可以选择的格式可以考虑: `TFRecord（Tensorflow）` , `recordIO（recordIO）` , `hdf5` , `pth` , `n5` , `lmdb` 等等(https://github.com/Lyken17/Efficient-PyTorch#data-loader)

- `TFRecord` :https://github.com/vahidk/tfrecord
- 借助 `lmdb` 数据库格式:
  - https://github.com/Fangyh09/Image2LMDB
  - https://blog.csdn.net/P_LarT/article/details/103208405
  - https://github.com/lartpang/PySODToolBox/blob/master/ForBigDataset/ImageFolder2LMDB.py

#### 预读取数据

- 预读取下一次迭代需要的数据

【参考】

- 如何给你PyTorch里的Dataloader打鸡血 - MKFMIKU的文章 - 知乎 https://zhuanlan.zhihu.com/p/66145913
- 给pytorch 读取数据加速 - 体hi的文章 - 知乎 https://zhuanlan.zhihu.com/p/72956595

#### 借助内存

- 直接载到内存里面, 或者把把内存映射成磁盘好了

【参考】

- 参见 https://zhuanlan.zhihu.com/p/66145913 的评论中 @雨宫夏一 的评论（额，似乎那条评论找不到了，不过大家可以搜一搜）

#### 借助固态

- 把读取速度慢的机械硬盘换成 NVME 固态吧～

【参考】

- 如何给你PyTorch里的Dataloader打鸡血 - MKFMIKU的文章 - 知乎 https://zhuanlan.zhihu.com/p/66145913

### 训练策略

***低精度训练***

- 在训练中使用低精度(`FP16`甚至`INT8`、二值网络、三值网络)表示取代原有精度(`FP32`)表示
  - 使用 Apex 的混合精度或者是PyTorch1.6开始提供的torch.cuda.amp模块来训练. 可以节约一定的显存并提速, 但是要小心一些不安全的操作如mean和sum:
    - `NVIDIA/Apex`:
      - https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729
      - https://github.com/nvidia/apex
      - Pytorch 安装 APEX 疑难杂症解决方案 - 陈瀚可的文章 - 知乎https://zhuanlan.zhihu.com/p/80386137
      - http://kevinlt.top/2018/09/14/mixed_precision_training/
    - `torch.cuda.amp`:
      - PyTorch的文档：https://pytorch.org/docs/stable/notes/amp_examples.html%3E

### 代码层面

- `torch.backends.cudnn.benchmark = True`
- Do numpy-like operations on the GPU wherever you can
- Free up memory using `del`
- Avoid unnecessary transfer of data from the GPU
- Use pinned memory, and use `non_blocking=True`to parallelize data transfer and GPU number crunching
  - 文档：https://pytorch.org/docs/stable/nn.html#torch.nn.Module.to
  - 关于 `non_blocking=True` 的设定的一些介绍：Pytorch有什么节省显存的小技巧？ - 陈瀚可的回答 - 知乎 https://www.zhihu.com/question/274635237/answer/756144739
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

- 有三AI:【杂谈】当前模型量化有哪些可用的开源工具?https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649037243&idx=1&sn=db2dc420c4d086fc99c7d8aada767484&chksm=8712a7c6b0652ed020872a97ea426aca1b06adf7571af3da6dac8ce991fd61001245e9bf6e9b&mpshare=1&scene=1&srcid=&sharer_sharetime=1576667804820&sharer_shareid=1d0dbdb37c6b95413d1d4fe7d61ed8f1&exportkey=A6g%2Fj50pMJYVXsedNyDVh9k%3D&pass_ticket=winxjBrzw0kHErbSri5yXS88yBx1a%2BAL9KKTG6Zt1MMS%2FeI2hpx%2BmeaLsrahnlOS#rd

#### 网络 inference 阶段 Conv 层和 BN 层融合

【参考】

- https://zhuanlan.zhihu.com/p/110552861
- PyTorch本身提供了类似的功能, 但是我没有使用过, 希望有朋友可以提供一些使用体会:https://pytorch.org/docs/1.3.0/quantization.html#torch.quantization.fuse_modules
- 网络inference阶段conv层和BN层的融合 - autocyz的文章 - 知乎 https://zhuanlan.zhihu.com/p/48005099

#### 多分支结构融合成单分支

- ACNet、RepVGG这种设计策略也很有意思
  - RepVGG|让你的ConVNet一卷到底，plain网络首次超过80%top1精度：https://mp.weixin.qq.com/s/M4Kspm6hO3W8fXT_JqoEhA

### 时间分析

- Python 的 `cProfile` 可以用来分析.(Python 自带了几个性能分析的模块: `profile` , `cProfile` 和 `hotshot` , 使用方法基本都差不多, 无非模块是纯 Python 还是用 C 写的)

### 项目推荐

- 基于 Pytorch 实现模型压缩(https://github.com/666DZY666/model-compression):
  - 量化:8/4/2 bits(dorefa)、三值/二值(twn/bnn/xnor-net)
  - 剪枝: 正常、规整、针对分组卷积结构的通道剪枝
  - 分组卷积结构
  - 针对特征二值量化的BN融合

### 扩展阅读

- pytorch dataloader数据加载占用了大部分时间, 各位大佬都是怎么解决的? - 知乎 https://www.zhihu.com/question/307282137
- 使用pytorch时, 训练集数据太多达到上千万张, Dataloader加载很慢怎么办? - 知乎 https://www.zhihu.com/question/356829360
- PyTorch 有哪些坑/bug? - 知乎 https://www.zhihu.com/question/67209417
- https://sagivtech.com/2017/09/19/optimizing-pytorch-training-code/
- 26秒单GPU训练CIFAR10, Jeff Dean也点赞的深度学习优化技巧 - 机器之心的文章 - 知乎 https://zhuanlan.zhihu.com/p/79020733
- 线上模型加入几个新特征训练后上线, tensorflow serving预测时间为什么比原来慢20多倍? - TzeSing的回答 - 知乎 https://www.zhihu.com/question/354086469/answer/894235805
- 相关资料 · 语雀 https://www.yuque.com/lart/gw5mta/bl3p3y
- ShuffleNetV2:https://arxiv.org/pdf/1807.11164.pdf
- 今天, 你的模型加速了吗? 这里有5个方法供你参考(附代码解析): https://mp.weixin.qq.com/s?__biz=MzI0ODcxODk5OA==&mid=2247511633&idx=2&sn=a5ab187c03dfeab4e64c85fc562d7c0d&chksm=e99e9da8dee914be3d713c41d5dedb7fcdc9982c8b027b5e9b84e31789913c5b2dd880210ead&mpshare=1&scene=1&srcid=&sharer_sharetime=1576934236399&sharer_shareid=1d0dbdb37c6b95413d1d4fe7d61ed8f1&exportkey=A%2B3SqYGse83qyFva%2BYSy3Ng%3D&pass_ticket=winxjBrzw0kHErbSri5yXS88yBx1a%2BAL9KKTG6Zt1MMS%2FeI2hpx%2BmeaLsrahnlOS#rd
- pytorch常见的坑汇总 - 郁振波的文章 - 知乎 https://zhuanlan.zhihu.com/p/77952356
- Pytorch 提速指南 - 云梦的文章 - 知乎 https://zhuanlan.zhihu.com/p/39752167



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
  - https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729
  - https://github.com/nvidia/apex
  - Pytorch 安装 APEX 疑难杂症解决方案 - 陈瀚可的文章 - 知乎https://zhuanlan.zhihu.com/p/80386137
  - http://kevinlt.top/2018/09/14/mixed_precision_training/
- `torch.cuda.amp`:
  - PyTorch的文档：https://pytorch.org/docs/stable/notes/amp_examples.html

### 对不需要反向传播的操作进行管理

- 对于不需要bp的forward, 如validation 请使用 `torch.no_grad` , 注意 `model.eval()` 不等于 `torch.no_grad()` , 请看如下讨论: ['model.eval()' vs 'with torch.no_grad()'](https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
- ~~将不需要更新的层的参数从优化器中排除~~将变量的 `requires_grad`设为 `False`, 让变量不参与梯度的后向传播(主要是为了减少不必要的梯度的显存占用)

### 显存清理

- `torch.cuda.empty_cache()`：这是`del`的进阶版, 使用`nvidia-smi`会发现显存有明显的变化. 但是训练时最大的显存占用似乎没变. 大家可以试试: [How can we release GPU memory cache?](https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530)
- 可以使用 `del` 删除不必要的中间变量, 或者使用 `replacing variables` 的形式来减少占用.

### 梯度累加

- 把一个 `batchsize=64` 分为两个32的batch, 两次forward以后, backward一次. 但会影响 `batchnorm` 等和 `batchsize` 相关的层. 在PyTorch的文档 https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation中提到了梯度累加与混合精度并用的例子.
- 这篇文章提到使用梯度累加技术实现对于分布式训练的加速：https://zhuanlan.zhihu.com/p/250471767

### 使用 `checkpoint` 技术

#### `torch.utils.checkpoint`

**这是更为通用的选择.**

【参考】

- https://blog.csdn.net/one_six_mix/article/details/93937091
- https://pytorch.org/docs/1.3.0/_modules/torch/utils/checkpoint.html#checkpoint

#### Training Deep Nets with Sublinear Memory Cost

方法来自论文: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174). 训练 CNN 时, Memory 主要的开销来自于储存用于计算 backward 的 activation, 一般的 workflow 是这样的

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/pytorch_memory_1.gif)

对于一个长度为 N 的 CNN, 需要 O(N) 的内存. 这篇论文给出了一个思路, 每隔 sqrt(N) 个 node 存一个 activation, 中需要的时候再算, 这样显存就从 O(N) 降到了 O(sqrt(N)).

![](images/pytorch_memory_2.gif)

对于越深的模型, 这个方法省的显存就越多, 且速度不会明显变慢.

![](https://gitee.com/lcai013/image_cdn/raw/master/notes_images/pytorch_memory_3.png)

实现的版本 https://github.com/Lyken17/pytorch-memonger

本文章首发在 极市计算机视觉技术社区

【参考】: Pytorch有什么节省内存(显存)的小技巧? - Lyken的回答 - 知乎 https://www.zhihu.com/question/274635237/answer/755102181

### 相关工具

- These codes can help you to detect your GPU memory during training with Pytorch. https://github.com/Oldpan/Pytorch-Memory-Utils
- Just less than nvidia-smi? https://github.com/wookayin/gpustat



### 参考资料

- Pytorch有什么节省内存(显存)的小技巧? - 郑哲东的回答 - 知乎 https://www.zhihu.com/question/274635237/answer/573633662
- 浅谈深度学习: 如何计算模型以及中间变量的显存占用大小 https://oldpan.me/archives/how-to-calculate-gpu-memory
- 如何在Pytorch中精细化利用显存 https://oldpan.me/archives/how-to-use-memory-pytorch
- Pytorch有什么节省显存的小技巧? - 陈瀚可的回答 - 知乎:https://www.zhihu.com/question/274635237/answer/756144739

## 4. 其他技巧

### 重现

#### 强制确定性操作

**PyTorch 1.7 新更新功能**

```
>>> import torch
>>> torch.set_deterministic(True)
```

【参考】:https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms



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

【参考】:[https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch%E9%9A%8F%E6%9C%BA%E7%A7%8D%E5%AD%90/](https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch随机种子/)





------

# 问题

## 1. 不同 python 版本下保存和加载模型的问题

报错信息：

```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 1124: ordinal not in range(128)
```

这个问题主要是由于模型在 Python2 下训练保存，但在 Python3 环境下加载出错，根源应该是 `pickle` 这个库的原因，即 `the pickle library in python which cannot correctly load the CNN model trained in python2.7 into python3.6`

具体查看：[UnicodeDecodeError#25](https://github.com/CSAILVision/places365/issues/25)

解决方法有两个：

1. 修改 pytorch 源码，`serialization.py` 代码中 lines 376-377 添加下列代码：

```python
_sys_info = pickle_module.load(f,encoding='latin1')
unpickler = pickle_module.Unpickler(f,encoding='latin1')
```

2. 加载模型代码如下所示：

```python
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
```

验证了第二种方法是可行的，解决方法链接：https://github.com/CSAILVision/places365/issues/25#issuecomment-333871990

------

## 2. `DataParallel` 使用网络的属性

定义好网络模型后，如果需要使用多 GPU，通常会采用 `torch.nn.DataParallel` ，但是这种情况下如果调用除方法 `forward` 外自定义的其他属性方法呢，解决方法如下：

来自 https://discuss.pytorch.org/t/fine-tuning-resnet-dataparallel-object-has-no-attribute-fc/14842

When using `DataParallel` your `nn.Module` will be in `.module`:

```python
model = Net()
model.fc
model_parallel = nn.DataParallel(model)
model_parallel.module.fc
```

------

## 3. 如何提升安装 pytorch 的速度

参考：

[conda安装Pytorch下载过慢解决办法(11月26日更新ubuntu下pytorch1.3安装方法)](https://blog.csdn.net/watermelon1123/article/details/88122020)

### conda

pytorch 的常见安装方法主要是 `pip` 或者 `conda` ，但都可能遇到下载速度过慢的情况，这里通常都是考虑采用国内源来代替默认 conda 源，比如清华的conda源：

清华conda源地址：https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

#### 添加清华源

通过命令行的方式添加：

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

为了保险起见，可以同时添加第三方 conda 源：

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

也可以在 conda 的配置文件中添加：

```
vim ~/.condarc
```

配置文件如下所示：

```
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
show_channel_urls: true
```

添加后，根据官方提升的安装命令是这样的：

```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

但这里需要删除最后的命令 `-c pytorch` ，这是制定采用默认的 conda 源，安装命令应该如下所示：

```
conda install pytorch torchvision cudatoolkit=9.0
```

#### 如何查看能不能用清华源加速你的pytorch安装

pytorch 安装中两个最大而且和版本相关的包，是**cudatoolkit-10.0**和**pytorch-1.4.0**。所以核心就是只要在清华源里找到这两个包，下载速度应该就没问题，地址分别是：

- [cudatoolkit安装包地址](https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/)
- [Pytorch安装包地址](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/)

### pip

用pip 安装 pytorch，速度可能会比较慢，参考下面这篇文章：

[【PyTorch学习笔记】一、安装GPU版本pytorch详细教程(避坑)](https://mp.weixin.qq.com/s/xlm0NkTZw_K1CCa-gys2Lg)

采用 `pip` 方式安装，添加阿里源，速度会提升很多，方法如下所示：

```
pip3 install https://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-win_amd64.whl -i http://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com
```

这是安装 `PyTorch` 1.1 版本， `cuda=9.0` ，python版本是 `3.6` ，其中关键命令是：

```
 -i http://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com
```

语句的意思是相当于用阿里云的 pip 源并且信任这个源，可以用。



------

