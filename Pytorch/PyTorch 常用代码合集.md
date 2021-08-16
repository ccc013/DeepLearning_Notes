# 数据加载和预处理




## 图片数据建模


在Pytorch中构建图片数据管道通常有两种方法。

1. 第一种是使用 torchvision 中的 `datasets.ImageFolder` 来读取图片然后用 DataLoader来并行加载。
1. 第二种是通过继承 `torch.utils.data.Dataset` 实现用户自定义读取逻辑然后用 DataLoade r来并行加载。

第二种方法是读取用户自定义数据集的通用方法，既可以读取图片数据集，也可以读取文本数据集。
​

### 采用 `dataset.ImageFolder` 加载的方法
加载自己的数据集，定义图片所在位置，如 `data/images` ，并且该文件夹下又包含 `train, val` 两个表示训练集和验证集的子文件夹。


```python
import torch 
from torch import nn
from torchvision import transforms,datasets 

# 数据增强方法，训练集会实现随机裁剪和水平翻转，然后进行归一化
# 验证集仅仅是裁剪和归一化，并不会做数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# 数据集所在文件夹
data_dir = ''
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```






### 可视化图片
可视化经过 `transforms` 处理的图片：


```python
# 图片展示的函数
def imshow(inp, title=None):
    """Imshow for Tensor."""
    # 逆转操作，从 tensor 变回 numpy 数组需要转换通道位置
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # 从归一化后变回原始图片
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
 
# 获取一个 batch 的训练数据
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```




## 文本数据建模
在torch中预处理文本数据一般使用 `torchtext`或者自定义Dataset，torchtext 功能非常强大，可以构建文本分类，序列标注，问答模型，机器翻译等NLP任务的数据集。
​

torchtext常见API一览

- torchtext.data.Example : 用来表示一个样本，数据和标签
- torchtext.vocab.Vocab: 词汇表，可以导入一些预训练词向量
- torchtext.data.Datasets: 数据集类，__getitem__返回 Example实例, torchtext.data.TabularDataset是其子类。
- torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）创建 Example时的 预处理，batch 时的一些处理操作。
- torchtext.data.Iterator: 迭代器，用来生成 batch
- torchtext.datasets: 包含了常见的数据集.

​

```python
import torch
import string,re
import torchtext

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20 

#分词方法
tokenizer = lambda x:re.sub('[%s]'%string.punctuation,"",x).split(" ")

#过滤掉低频词
def filterLowFreqWords(arr,vocab):
    arr = [[x if x<MAX_WORDS else 0 for x in example] 
           for example in arr]
    return arr

#1,定义各个字段的预处理方法
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True, 
                  fix_length=MAX_LEN,postprocessing = filterLowFreqWords)

LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

#2,构建表格型dataset
#torchtext.data.TabularDataset可读取csv,tsv,json等格式
ds_train, ds_valid = torchtext.data.TabularDataset.splits(
        path='./data/imdb', train='train.tsv',test='test.tsv', format='tsv',
        fields=[('label', LABEL), ('text', TEXT)],skip_header = False)

#3,构建词典
TEXT.build_vocab(ds_train)

#4,构建数据管道迭代器
train_iter, valid_iter = torchtext.data.Iterator.splits(
        (ds_train, ds_valid),  sort_within_batch=True,sort_key=lambda x: len(x.text),
        batch_sizes=(BATCH_SIZE,BATCH_SIZE))


```
```python
#查看example信息
print(ds_train[0].text)
print(ds_train[0].label)
```
输出结果
```python
['it', 'really', 'boggles', 'my', 'mind', 'when', 'someone', 'comes', 'across', 'a', 'movie', 'like', 'this', 'and', 'claims', 'it', 'to', 'be', 'one', 'of', 'the', 'worst', 'slasher', 'films', 'out', 'there', 'this', 'is', 'by', 'far', 'not', 'one', 'of', 'the', 'worst', 'out', 'there', 'still', 'not', 'a', 'good', 'movie', 'but', 'not', 'the', 'worst', 'nonetheless', 'go', 'see', 'something', 'like', 'death', 'nurse', 'or', 'blood', 'lake', 'and', 'then', 'come', 'back', 'to', 'me', 'and', 'tell', 'me', 'if', 'you', 'think', 'the', 'night', 'brings', 'charlie', 'is', 'the', 'worst', 'the', 'film', 'has', 'decent', 'camera', 'work', 'and', 'editing', 'which', 'is', 'way', 'more', 'than', 'i', 'can', 'say', 'for', 'many', 'more', 'extremely', 'obscure', 'slasher', 'filmsbr', 'br', 'the', 'film', 'doesnt', 'deliver', 'on', 'the', 'onscreen', 'deaths', 'theres', 'one', 'death', 'where', 'you', 'see', 'his', 'pruning', 'saw', 'rip', 'into', 'a', 'neck', 'but', 'all', 'other', 'deaths', 'are', 'hardly', 'interesting', 'but', 'the', 'lack', 'of', 'onscreen', 'graphic', 'violence', 'doesnt', 'mean', 'this', 'isnt', 'a', 'slasher', 'film', 'just', 'a', 'bad', 'onebr', 'br', 'the', 'film', 'was', 'obviously', 'intended', 'not', 'to', 'be', 'taken', 'too', 'seriously', 'the', 'film', 'came', 'in', 'at', 'the', 'end', 'of', 'the', 'second', 'slasher', 'cycle', 'so', 'it', 'certainly', 'was', 'a', 'reflection', 'on', 'traditional', 'slasher', 'elements', 'done', 'in', 'a', 'tongue', 'in', 'cheek', 'way', 'for', 'example', 'after', 'a', 'kill', 'charlie', 'goes', 'to', 'the', 'towns', 'welcome', 'sign', 'and', 'marks', 'the', 'population', 'down', 'one', 'less', 'this', 'is', 'something', 'that', 'can', 'only', 'get', 'a', 'laughbr', 'br', 'if', 'youre', 'into', 'slasher', 'films', 'definitely', 'give', 'this', 'film', 'a', 'watch', 'it', 'is', 'slightly', 'different', 'than', 'your', 'usual', 'slasher', 'film', 'with', 'possibility', 'of', 'two', 'killers', 'but', 'not', 'by', 'much', 'the', 'comedy', 'of', 'the', 'movie', 'is', 'pretty', 'much', 'telling', 'the', 'audience', 'to', 'relax', 'and', 'not', 'take', 'the', 'movie', 'so', 'god', 'darn', 'serious', 'you', 'may', 'forget', 'the', 'movie', 'you', 'may', 'remember', 'it', 'ill', 'remember', 'it', 'because', 'i', 'love', 'the', 'name']
0
```
```python
# 查看词典信息
print(len(TEXT.vocab))

#itos: index to string
print(TEXT.vocab.itos[0]) 
print(TEXT.vocab.itos[1]) 

#stoi: string to index
print(TEXT.vocab.stoi['<unk>']) #unknown 未知词
print(TEXT.vocab.stoi['<pad>']) #padding  填充


#freqs: 词频
print(TEXT.vocab.freqs['<unk>']) 
print(TEXT.vocab.freqs['a']) 
print(TEXT.vocab.freqs['good']) 
```
```python
108197
<unk>
<pad>
0
1
0
129453
11457

```
```python
# 查看数据管道信息
# 注意有坑：text第0维是句子长度
for batch in train_iter:
    features = batch.text
    labels = batch.label
    print(features)
    print(features.shape)
    print(labels)
    break

```
```python
tensor([[  17,   31,  148,  ...,   54,   11,  201],
        [   2,    2,  904,  ...,  335,    7,  109],
        [1371, 1737,   44,  ...,  806,    2,   11],
        ...,
        [   6,    5,   62,  ...,    1,    1,    1],
        [ 170,    0,   27,  ...,    1,    1,    1],
        [  15,    0,   45,  ...,    1,    1,    1]])
torch.Size([200, 20])
tensor([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0])
```
```python
# 将数据管道组织成torch.utils.data.DataLoader相似的features,label输出形式
class DataLoader:
    def __init__(self,data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        # 注意：此处调整features为 batch first，并调整label的shape和dtype
        for batch in self.data_iter:
            yield(torch.transpose(batch.text,0,1),
                  torch.unsqueeze(batch.label.float(),dim = 1))
    
dl_train = DataLoader(train_iter)
dl_valid = DataLoader(valid_iter)


```




## 时间序列数据建模
通过继承 `torch.utils.data.Dataset`实现自定义时间序列数据集。
`torch.utils.data.Dataset`是一个抽象类，用户想要加载自定义的数据只需要继承这个类，并且覆写其中的两个方法即可：

- __len__:实现len(dataset)返回整个数据集的大小。
- __getitem__:用来获取一些索引的数据，使dataset[i]返回数据集中第i个样本。

不覆写这两个方法会直接返回错误。


```python
import torch 
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset


#用某日前8天窗口数据作为输入预测该日数据
WINDOW_SIZE = 8

class Covid19Dataset(Dataset):
        
    def __len__(self):
        return len(dfdiff) - WINDOW_SIZE
    
    def __getitem__(self,i):
        x = dfdiff.loc[i:i+WINDOW_SIZE-1,:]
        feature = torch.tensor(x.values)
        y = dfdiff.loc[i+WINDOW_SIZE,:]
        label = torch.tensor(y.values)
        return (feature,label)
    
ds_train = Covid19Dataset()

#数据较小，可以将全部训练数据放入到一个batch中，提升性能
dl_train = DataLoader(ds_train,batch_size = 38)

```





---

# 模型定义


## 载入预训练模型


```python
if os.path.isfile(restore_model):
    t1 = time.time()
    checkpoint = torch.load(restore_model)
    logger.info('checkpoint keys: {}'.format(checkpoint.keys()))
    assert 'state_dict' in checkpoint, Exception('[state_dict] not in checkpoint')
    for k in checkpoint.keys():
        if k != 'state_dict':
            logger.info('{}={}'.format(k, checkpoint[k]))
    args.start_epoch = checkpoint.get('epoch', 0)
    net.load_state_dict(checkpoint['state_dict'])
    logger.info('=> load checkpoint {}, start_epoch={}, time={:.4f}'.format(
                args.restore, args.start_epoch, time.time() - t1))
else:
    if restore_model is None or restore_model == '':
        restore_model = 'pretrained_model'
    logger.info('{} not exist. So train the model from scratch.'.format(restore_model))
```


## 加载 ImageNet 预训练模型


```python
def load_state_dict(model, model_root):
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict()  # remove all 'group' string
    for k, v in own_state_old.items():
        if 'fc' in k:
            continue
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

        # print('layer {}'.format(k))

    state_dict = torch.load(model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            if 'fc' in name:
                continue
            if 'layer4.0' in name or 'layer4.1' in name or 'layer4.2' in name:
                name = 'layer4_' + name.split(".", 2)[1] + '.0.' + name.split(".", 2)[2]
            else:
                print(own_state.keys())
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

```


## 定义模型
使用Pytorch通常有三种方式构建模型：

- 使用 `nn.Sequential`按层顺序构建模型
- 继承 `nn.Module`基类构建自定义模型
- 继承 `nn.Module`基类构建模型并辅助应用模型容器进行封装。

​

### 方法 1：nn.Sequential
第一种方法，使用 `nn.Sequential`按层顺序构建模型例子如下所示：
```python
def create_net():
    net = nn.Sequential()
    net.add_module("linear1",nn.Linear(15,20))
    net.add_module("relu1",nn.ReLU())
    net.add_module("linear2",nn.Linear(20,15))
    net.add_module("relu2",nn.ReLU())
    net.add_module("linear3",nn.Linear(15,1))
    net.add_module("sigmoid",nn.Sigmoid())
    return net
    
net = create_net()
print(net)
```
输出结果如下：
```python
Sequential(
  (linear1): Linear(in_features=15, out_features=20, bias=True)
  (relu1): ReLU()
  (linear2): Linear(in_features=20, out_features=15, bias=True)
  (relu2): ReLU()
  (linear3): Linear(in_features=15, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
```
调用 `torchkeras.summary`，给定一个输入的大小，可以查看输出网络模型的参数和大小
```python
from torchkeras import summary
summary(net,input_shape=(15,))
```
输出结果：
```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                   [-1, 20]             320
              ReLU-2                   [-1, 20]               0
            Linear-3                   [-1, 15]             315
              ReLU-4                   [-1, 15]               0
            Linear-5                    [-1, 1]              16
           Sigmoid-6                    [-1, 1]               0
================================================================
Total params: 651
Trainable params: 651
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.000057
Forward/backward pass size (MB): 0.000549
Params size (MB): 0.002483
Estimated Total Size (MB): 0.003090
----------------------------------------------------------------
```
​

### 方法 2：继承nn.Module基类构建自定义模型
```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
        self.dropout = nn.Dropout2d(p = 0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y
        
net = Net()
print(net)

```
输出结果如下：
```python
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
  (dropout): Dropout2d(p=0.1, inplace=False)
  (adaptive_pool): AdaptiveMaxPool2d(output_size=(1, 1))
  (flatten): Flatten()
  (linear1): Linear(in_features=64, out_features=32, bias=True)
  (relu): ReLU()
  (linear2): Linear(in_features=32, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)

```
调用 `torchkeras.summary`，给定一个输入的大小，可以查看输出网络模型的参数和大小
```python
import torchkeras
torchkeras.summary(net,input_shape= (3,32,32))
```
输出结果：
```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 30, 30]             896
         MaxPool2d-2           [-1, 32, 15, 15]               0
            Conv2d-3           [-1, 64, 11, 11]          51,264
         MaxPool2d-4             [-1, 64, 5, 5]               0
         Dropout2d-5             [-1, 64, 5, 5]               0
 AdaptiveMaxPool2d-6             [-1, 64, 1, 1]               0
           Flatten-7                   [-1, 64]               0
            Linear-8                   [-1, 32]           2,080
              ReLU-9                   [-1, 32]               0
           Linear-10                    [-1, 1]              33
          Sigmoid-11                    [-1, 1]               0
================================================================
Total params: 54,273
Trainable params: 54,273
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.011719
Forward/backward pass size (MB): 0.359634
Params size (MB): 0.207035
Estimated Total Size (MB): 0.578388
----------------------------------------------------------------

```
​

### 方法 3：继承 `nn.Module`基类构建模型并辅助应用模型容器进行封装。
```python
import torch
from torch import nn 
from torchkeras import LightModel,summary 


torch.random.seed()
import torch
from torch import nn 

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        #设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_embeddings = MAX_WORDS,embedding_dim = 3,padding_idx = 1)
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1",nn.Conv1d(in_channels = 3,out_channels = 16,kernel_size = 5))
        self.conv.add_module("pool_1",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_1",nn.ReLU())
        self.conv.add_module("conv_2",nn.Conv1d(in_channels = 16,out_channels = 128,kernel_size = 2))
        self.conv.add_module("pool_2",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_2",nn.ReLU())
        
        self.dense = nn.Sequential()
        self.dense.add_module("flatten",nn.Flatten())
        self.dense.add_module("linear",nn.Linear(6144,1))
        self.dense.add_module("sigmoid",nn.Sigmoid())
        
    def forward(self,x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)
        return y
        

net = Net()
print(net)

summary(net, input_shape = (200,),input_dtype = torch.LongTensor)
```
打印结果
```python
Net(
  (embedding): Embedding(10000, 3, padding_idx=1)
  (conv): Sequential(
    (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
    (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_1): ReLU()
    (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
    (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (relu_2): ReLU()
  )
  (dense): Sequential(
    (flatten): Flatten()
    (linear): Linear(in_features=6144, out_features=1, bias=True)
    (sigmoid): Sigmoid()
  )
)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         Embedding-1               [-1, 200, 3]          30,000
            Conv1d-2              [-1, 16, 196]             256
         MaxPool1d-3               [-1, 16, 98]               0
              ReLU-4               [-1, 16, 98]               0
            Conv1d-5              [-1, 128, 97]           4,224
         MaxPool1d-6              [-1, 128, 48]               0
              ReLU-7              [-1, 128, 48]               0
           Flatten-8                 [-1, 6144]               0
            Linear-9                    [-1, 1]           6,145
          Sigmoid-10                    [-1, 1]               0
================================================================
Total params: 40,625
Trainable params: 40,625
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.000763
Forward/backward pass size (MB): 0.287796
Params size (MB): 0.154972
Estimated Total Size (MB): 0.443531
----------------------------------------------------------------

```
​

​

​


---

## 保存模型
Pytorch 有两种保存模型的方式，都是通过调用pickle序列化方法实现的。
第一种方法只保存模型参数。
第二种方法保存完整模型。
推荐使用第一种，第二种方法可能在切换设备和目录的时候出现各种问题。
​

首先是第一种保存模型参数的做法：


```python
import torch
# model 定义好的网络模型
if use_cuda and torch.cuda.is_available():
    # 使用 gpu
	state_dict = model.module.state_dict()
else:
    state_dict = model.state_dict()
for key in state_dict.keys():
    state_dict[key] = state_dict[key].cpu()
save_checkpoint_file = os.path.join(save_model_dir, '%03d_%.4f.ckpt' % (epoch + 1, val_acc))
save_dict = {'epoch': (epoch + 1),
             'save_dir': save_model_dir,
             'state_dict': state_dict,
             'lr': optimizer.param_groups[-1]['lr'],
             'val_acc': val_acc
            }
torch.save(save_dict, save_checkpoint_file)

```


如果仅想保留最好的模型: 


```python
# 删除指定的旧模型，仅保留一个或少数几个效果最好的模型的做法
old_best_models = glob.glob(os.path.join(save_model_dir, 'best_model_*.ckpt'))
for old_model in old_best_models:
    os.remove(old_model)
save_model_path = os.path.join(save_model_dir, 'best_model_{:.4f}.ckpt'.format(best_acc))
shutil.copyfile(save_checkpoint_file, save_model_path)
```




第二种，保存完整模型
```python
torch.save(net, './data/net_model.pkl')
net_loaded = torch.load('./data/net_model.pkl')
net_loaded(torch.tensor(x_test[0:10]).float()).data
```









---

# 模型训练和预测
## 
## 训练模型
有3类典型的训练循环代码风格：脚本形式训练循环，函数形式训练循环，类形式训练循环。
​

### 通用的训练写法


这里介绍一种较通用的脚本形式。
```python
from sklearn.metrics import accuracy_score

# 定义 loss，优化函数和评价标准
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.01)
metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy()>0.5)
metric_name = "accuracy"

# 定义训练的 epoch
epochs = 10
log_step_freq = 30

dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name]) 
print("Start Training...")
nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("=========="*8 + "%s"%nowtime)
# 开始训练
for epoch in range(1,epochs+1):  

    # 1，训练循环-------------------------------------------------
    net.train()
    loss_sum = 0.0
    metric_sum = 0.0
    step = 1
    
    for step, (features,labels) in enumerate(dl_train, 1):
    
        # 梯度清零
        optimizer.zero_grad()

        # 正向传播求损失
        predictions = net(features)
        loss = loss_func(predictions,labels)
        metric = metric_func(predictions,labels)
        
        # 反向传播求梯度
        loss.backward()
        optimizer.step()

        # 打印batch级别日志
        loss_sum += loss.item()
        metric_sum += metric.item()
        if step%log_step_freq == 0:   
            print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                  (step, loss_sum/step, metric_sum/step))
            
    # 2，验证循环-------------------------------------------------
    net.eval()
    val_loss_sum = 0.0
    val_metric_sum = 0.0
    val_step = 1

    for val_step, (features,labels) in enumerate(dl_valid, 1):
        # 关闭梯度计算
        with torch.no_grad():
            predictions = net(features)
            val_loss = loss_func(predictions,labels)
            val_metric = metric_func(predictions,labels)
        val_loss_sum += val_loss.item()
        val_metric_sum += val_metric.item()

    # 3，记录日志-------------------------------------------------
    info = (epoch, loss_sum/step, metric_sum/step, 
            val_loss_sum/val_step, val_metric_sum/val_step)
    dfhistory.loc[epoch-1] = info
    
    # 打印epoch级别日志
    print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
          "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f") 
          %info)
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
        
print('Finished Training...')
```
根据保存在 dataframe 来绘制训练 loss 曲线
```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory,"loss")
```


### 类形式的训练循环
利用Pytorch-Lightning定义了一个高阶的模型接口LightModel, 封装在torchkeras中, 可以非常方便地训练模型。
```python
import pytorch_lightning as pl 
from torchkeras import LightModel 

class Model(LightModel):
    
    #loss,and optional metrics
    def shared_step(self,batch)->dict:
        x, y = batch
        prediction = self(x)
        loss = nn.BCELoss()(prediction,y)
        preds = torch.where(prediction>0.5,torch.ones_like(prediction),torch.zeros_like(prediction))
        acc = pl.metrics.functional.accuracy(preds, y)
        dic = {"loss":loss,"accuracy":acc} 
        return dic
    
    #optimizer,and optional lr_scheduler
    def configure_optimizers(self):
        optimizer= torch.optim.Adagrad(self.parameters(),lr = 0.02)
        return optimizer


```
```python
pl.seed_everything(1234)
net = Net()
model = Model(net)

ckpt_cb = pl.callbacks.ModelCheckpoint(monitor='val_loss')

# set gpus=0 will use cpu，
# set gpus=1 will use 1 gpu
# set gpus=2 will use 2gpus 
# set gpus = -1 will use all gpus 
# you can also set gpus = [0,1] to use the  given gpus
# you can even set tpu_cores=2 to use two tpus 

trainer = pl.Trainer(max_epochs=20,gpus = 0, callbacks=[ckpt_cb]) 
trainer.fit(model,dl_train,dl_valid)

```
```python
================================================================================2021-01-16 21:47:29
epoch =  0
{'val_loss': 0.6834630966186523, 'val_accuracy': 0.5546000003814697}
{'accuracy': 0.5224003791809082, 'loss': 0.7246873378753662}

================================================================================2021-01-16 21:48:07
epoch =  1
{'val_loss': 0.6371415257453918, 'val_accuracy': 0.63319993019104}
{'accuracy': 0.6110503673553467, 'loss': 0.6552867889404297}

================================================================================2021-01-16 21:48:50
epoch =  2
{'val_loss': 0.5896139740943909, 'val_accuracy': 0.6798002123832703}
{'accuracy': 0.6910000443458557, 'loss': 0.5874115824699402}

================================================================================2021-01-16 21:49:32
epoch =  3
{'val_loss': 0.5726749300956726, 'val_accuracy': 0.6971999406814575}
{'accuracy': 0.7391000390052795, 'loss': 0.5251786112785339}

================================================================================2021-01-16 21:50:13
epoch =  4
{'val_loss': 0.5328916311264038, 'val_accuracy': 0.7326000332832336}
{'accuracy': 0.7705488801002502, 'loss': 0.4773417115211487}

================================================================================2021-01-16 21:50:54
epoch =  5
{'val_loss': 0.5194208025932312, 'val_accuracy': 0.7413997650146484}
{'accuracy': 0.7968998551368713, 'loss': 0.43944093585014343}

================================================================================2021-01-16 21:51:35
epoch =  6
{'val_loss': 0.5199333429336548, 'val_accuracy': 0.7429998517036438}
{'accuracy': 0.8130489587783813, 'loss': 0.4102325737476349}

================================================================================2021-01-16 21:52:16
epoch =  7
{'val_loss': 0.5124538540840149, 'val_accuracy': 0.7517998814582825}
{'accuracy': 0.8314500451087952, 'loss': 0.3849221169948578}

================================================================================2021-01-16 21:52:58
epoch =  8
{'val_loss': 0.510671079158783, 'val_accuracy': 0.7554002404212952}
{'accuracy': 0.8438503742218018, 'loss': 0.3616768419742584}

================================================================================2021-01-16 21:53:39
epoch =  9
{'val_loss': 0.5184627771377563, 'val_accuracy': 0.7530001997947693}
{'accuracy': 0.8568001985549927, 'loss': 0.34138554334640503}

================================================================================2021-01-16 21:54:20
epoch =  10
{'val_loss': 0.5105863809585571, 'val_accuracy': 0.7580001354217529}
{'accuracy': 0.865899920463562, 'loss': 0.32265418767929077}

================================================================================2021-01-16 21:55:02
epoch =  11
{'val_loss': 0.5222727656364441, 'val_accuracy': 0.7586002349853516}
{'accuracy': 0.8747013211250305, 'loss': 0.306064248085022}

================================================================================2021-01-16 21:55:43
epoch =  12
{'val_loss': 0.5208917856216431, 'val_accuracy': 0.7597998976707458}
{'accuracy': 0.8820013403892517, 'loss': 0.29068493843078613}

================================================================================2021-01-16 21:56:24
epoch =  13
{'val_loss': 0.5236031413078308, 'val_accuracy': 0.7603999376296997}
{'accuracy': 0.889351487159729, 'loss': 0.2765159606933594}

================================================================================2021-01-16 21:57:04
epoch =  14
{'val_loss': 0.5428195595741272, 'val_accuracy': 0.7572000622749329}
{'accuracy': 0.8975020051002502, 'loss': 0.26261812448501587}

================================================================================2021-01-16 21:57:45
epoch =  15
{'val_loss': 0.5340956449508667, 'val_accuracy': 0.7602002024650574}
{'accuracy': 0.9049026966094971, 'loss': 0.25028231739997864}

================================================================================2021-01-16 21:58:25
epoch =  16
{'val_loss': 0.5380828380584717, 'val_accuracy': 0.7612000107765198}
{'accuracy': 0.9085531234741211, 'loss': 0.23980091512203217}

================================================================================2021-01-16 21:59:05
epoch =  17
{'val_loss': 0.5447139739990234, 'val_accuracy': 0.7638000249862671}
{'accuracy': 0.9168024659156799, 'loss': 0.22760336101055145}

================================================================================2021-01-16 21:59:45
epoch =  18
{'val_loss': 0.5505074858665466, 'val_accuracy': 0.7636001110076904}
{'accuracy': 0.921653687953949, 'loss': 0.21746191382408142}

================================================================================2021-01-16 22:00:26
epoch =  19
{'val_loss': 0.5615255236625671, 'val_accuracy': 0.7634001970291138}
{'accuracy': 0.9263033270835876, 'loss': 0.2077799290418625}

```





---

# 结果分析
## 可视化feature maps
参考：

- [PyTorch提取中间层特征？](https://www.zhihu.com/question/68384370)
- [pytorch使用hook打印中间特征图、计算网络算力等](https://zhuanlan.zhihu.com/p/73868323)



特征图打印，利用 hook 机制


```python
import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

def viz(module, input):
    x = input[0][0]
    #最多显示4张图
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i+1)
        plt.imshow(x[i].cpu())
    plt.show()


import cv2
import numpy as np
def main():
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet18(pretrained=True).to(device)
    for name, m in model.named_modules():
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)
    img = cv2.imread('./cat.jpeg')
    img = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(img)

if __name__ == '__main__':
    main()
```









---

# 参考


- [PyTorch Cookbook（常用代码段整理合集）](https://zhuanlan.zhihu.com/p/59205847?)
- [https://towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051](https://towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051)
- [https://github.com/zxdefying/pytorch_tricks](https://github.com/zxdefying/pytorch_tricks)
- [[深度学习框架]PyTorch常用代码段](https://zhuanlan.zhihu.com/p/104019160)
- [https://github.com/lyhue1991/eat_pytorch_in_20_days](https://github.com/lyhue1991/eat_pytorch_in_20_days)
