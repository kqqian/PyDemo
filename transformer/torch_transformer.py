import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math, copy


class SublayerConnection(nn.Module):
    """
    Add & Norm
    """
    def __init__(self, size, dropout=0.1):
        """
        初始化方法，这里做LN的大小的初始化以及dropout
        :param size: LN大小
        :param dropout: dropout比例
        """
        super(SublayerConnection, self).__init__()
        # 1) 做layerNorm
        self.layer_norm = LayerNorm(size)
        # 2) 做dropout，可做可不做
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        Add & Norm
        :param x: 上一层的输入，这里就是Self-Attention的输入
        :param sublayer: 上一层，这里就是Self-Attention, sublayer(x)返回上一层的输出
        :return: 返回Norm(Add)后的结果
        """
        return self.dropout(self.layer_norm(x + sublayer(x)))


class LayerNorm(nn.Module):
    """
    LN实现代码
    """
    def __init__(self, features, eps=1e-6):
        """
        初始化方法
        :param features: 特征大小，这里是Self-Attention的x的大小
        :param eps: 归一化分布，避免为0参数
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 均值、方差
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.a_2*(x - mean)/(std+self.eps)+self.b_2


class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        """
        初始化
        :param head: 头数，默认8
        :param d_model: 输入的数据特征维度，512
        :param dropout: dropout
        """
        super(MultiHeadAttention, self).__init__()
        # 需要注意，d_model需要被head整除
        assert (d_model % head == 0)
        self.d_model = d_model
        self.d_k = d_model // head
        self.head = head
        # 线性层，用于将x转换成对应的q,k,v
        self.linear_query = nn.Linear(self.d_model, self.d_model)
        self.linear_key = nn.Linear(self.d_model, self.d_model)
        self.linear_value = nn.Linear(self.d_model, self.d_model)
        # 自注意力机制，QKV同源，线性变换
        # 输出时， Z维度的降低到模型的维度
        self.linear_out = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self,query, key, value, mask=None):
        """
        前向传播，通常数据会放到这个里面
        其中QKV都来源于X，X的大小就是一个向量的大小。
        比如一个单词向量就是[1,512]维度，通过QK点乘计算得到attention向量A，然后和V相乘后得到新的X向量，这个X向量具有更多的语义信息
        :param query: 查询集 Q，
        :param key: K
        :param value: V
        :param mask: 掩码mask
        :return:
        """
        # 训练模型都是按批次训练的，所以每个batch中的X都需要分成多个头
        n_batch = query.size(0)
        # 对 X 切分成多头
        # 首先用对应的linear，首先用Query Linear，从X中获取Q向量，得到的是(n_batch, 32, 512)大小的向量，然后进行切分成8头
        # Key Linear， Value Linear也是相同的，分成8份，也就是8头，X中每个数据x都会有一个qkv向量，所以这里的大小是n_batch, -1 8 64,这里的-1是计算得到的，这里是32，n_batch就是一组词向量（一句话的词向量）
        # 所以这里本质上QKV都是同一个同理，都具有一句话的含义
        # 如果这里的X是一句话，那么n_batch就是一句话的分词的长度，-1对应的是
        # query == key == value，所以在调用forward的地方，query=key=value都输入的是x
        # todo 这里的n_batch, X到底是多大的？X是一个词？还是一个向量？
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1,2) # view-> (n_batch, 32, 8, 64)， 8*64=512, 这里的32是一个视频的32帧；如果这里是文本信息，那么这里的维度就是(n_batch, 1, 8, 64)
        # query = self.linear_query(query).view(n_batch, self.head, -1, self.d_k) # 上一行的另外一种写法
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1,2) # (n_batch, 32, 8, 64)
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transponse(1,2) # (n_batch, 32, 8, 64)

        # 自注意力计算，得到具有更多信息的x向量
        x, self.attn = self.self_attention(query, key, value, dropout=self.dropout, mask=mask)
        # 变成三维，或者说是concat head
        # 注：transpose后，需要执行一些contiguous才能执行view操作
        # 这里的32是视频任务每一个视频有32帧。
        x = x.transpose(1,2).contiguous().view(n_batch, -1, self.head * self.d_k) # (n_batch, 32, 512)

        return self.linear_out(x)


    def self_attention(query, key, value, dropout=None, mask=None):
        # 获取queries的特征向量大小，用于softmax前，对注意力矩阵处理，避免softmax之后的概率差异很大
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose([-2, -1])) / torch.sqrt(d_k)
        # 掩码，解码器部分用到
        if mask is not None:
            # mask.cuda() #本项目用CPU，如果需要用到cuda，可以进行转换
            scores = scores.masked_fill(mask == 0, -1e9)
        # softmax
        self_attn = F.softmax(scores, dim=-1)
        # dropout
        if dropout is not None:
            self_attn = dropout(self_attn)
        # 注意力A和value相乘，得到注意力结果z；这里还返回了注意力的分数值
        return torch.matmul(self_attn, value), self_attn


class PositionalEmbedding(nn.Module):
    """
    实现cos/sin的位置编码
    """
    def __init__(self, max_seq_len, dim, dropout=0.1):
        """
        初始化函数，主要是初始化一个固定的cos/sin数组，总大小为[seq_len, 1, d_model]，其中d_model=512是词向量大小
        PE(pos, 2i) = sin(pos / 10000*(2i/d_model)))
        PE(pos, 2i+1) = cos(pos / 10000*(2i/d_model)))
        :param max_seq_len:
        """
        # init
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_seq_len, dim)
        # pos,生成一个从0开始的序列，得到[seq_len]，增加一个维度，得到[seq_len, 1]
        pos = torch.arange(0, max_seq_len, dtype=float).unsqueeze(1) #转换后，后面对div_term运算才会进行广播
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=float)) * -(math.log(10000) / dim))
        # sin 0::2-> 0, 2, 4...偶数位，cos 1::2-> 1, 3, 5...奇数位
        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)

        # print(pe.size()) # 得到的大小[10, 512]
        # 为了匹配后面positional中的词向量的维度[1, 512]
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, embd, step=None):
        """
        对x的编码embd进行位置编码
        :param embd: [seq_len, n_batch, d_model], pe: [max_seq_len, 1, d_model], 两者两加后，在n_batch维度上广播，实际叠加中seq_len=max_seq_len(取对应seq length大小)
        :param step:  是对一个embedding编码，还是一个序列编码
        :return:
        """
        embd = embd / math.sqrt(self.dim)
        seq_len = embd.size(0)
        print(seq_len)
        if step is not None:
            embd = embd + self.pe[step]
        else:
            embd = embd + self.pe[:seq_len]
        embd = self.dropout(embd)
        return embd


class PositionWiseFeedForward(nn.Module):
    """
    FFN前向传播网络
    w2(relu(w1(layer\_norm(x))+b1)) + b2
    """
    def __init__(self, d_model, d_ff, dropout_1=0.1, dropout_2=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff) # d_model -> d_ff
        self.w2 = nn.Linear(d_ff, d_model) # d_ff -> d_model
        self.dropout_1 = nn.Dropout(dropout_1) # 内层dropout
        self.dropout_2 = nn.Dropout(dropout_2) # 外层dropout
        self.relu = nn.ReLU() # 内层relu变换
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) # 通常需要对x做一下标准化操作

    def forward(self, x):
        """
        实现公式：w2(relu(w1(layer\_norm(x))+b1)) + b2，其中b1，b2通常直接设计到w中，x会增加一个维度，值为1
        :param x:
        :return:
        """
        inner = self.dropout_1(self.relu(self.w1(self.layer_norm(x))))
        outer = self.dropout_2(self.w2(inner))
        return outer

class Generator(nn.Module):
    """
    输出层：Linear + Softmax
    """
    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 词向量大小
        :param vocab_size: 整个词表每个词的概率值，大小为vocabulary size
        """
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)

def subsequence_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.tensor(mask == 0)


def clones(obj ,n):
    return nn.ModuleList([copy.deepcopy(obj) for _ in range(n)])

class EncoderLayer(nn.Module):
    """
    编码器层
    """
    def __init__(self, attn, feed_forward, size, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        # 1)首先经过self-attention + Add & Norm
        # sublayer_connection： 残差神经网络，它的输入是上一层的输出以及x，上一层的输出是自注意力层，也就是Self-Attention层
        l1 = self.sublayers[0](x, lambda x : self.attn(x, x, x, mask))
        # 2)再经过 ffn + add & norm
        l2 = self.sublayers[1](x, self.feed_forward)
        return l2

class Encoder(nn.Module):
    """
    编码器，transformer默认有6x Encoder Layer
    """
    def __init__(self, encoderLayer, n):
        super(Encoder, self).__init__()
        # encl = EncoderLayer(attn, ffn, size, dropout)
        self.encs = clones(encoderLayer, n)

    def forward(self, x, src_mask):
        for enc in self.encs:
            x = enc(x, src_mask)
        return x

class DecoderLayer(nn.Module):
    """
    解码层
    """
    def __init__(self, attn, size, dropout, feed_forward, sublayer_num):
        super(DecoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        # clone 3层 残差网络
        self.sublayers = clones(SublayerConnection(size,dropout), sublayer_num)

    # def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_memory=None):
    def forward(self, x, memory, src_mask, trg_mask, r2l_memory=None, r2l_trg_mask=None):
        """
        编码器单个模块的实现
        这里还涉及到双向解码器: 正向解码器还去接收反向解码器的内容
        :param x:
        :param memory: KV，编码器层得到
        :param trg_mask: 目标掩码
        :param r2l_memory: 反向解码器KV（反向掩码attention输出的KV向量）
        :param r2l_trg_mask: 反向解码器mask掩码层，就是和正向解码器相反的方向（反向时，就是从后往前输入序列）
        L2RDecoder: 正向解码器， R2LDecoder: 反向解码器
        :return:
        """
        # Masked Self-Attention, Add & Norm
        l1 = self.sublayers[0](x, lambda x : self.attn(x, x, x, trg_mask))
        # Self-Attention, Add & Norm
        l2 = self.sublayers[1](l1, lambda x : self.attn(x, memory, memory, src_mask))
        # 双向解码器，正向解码器还去接收反向解码器的内容
        if r2l_memory is not None:
            l2 = self.sublayers[2](l2, lambda x : self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))
        # Feed Forward, Add & Norm
        l3 = self.sublayers[-1](l2, self.feed_forward)
        return l3


class R2LDecoder(nn.Module):
    def __init__(self):
        super(R2LDecoder, self).__init__()

    def forward(self):
        print(1)

class L2RDecoder(nn.Module):
    def __init__(self):
        super(L2RDecoder, self).__init__()
        


if __name__ == "__main__":
    rs = clones(SublayerConnection([1,1]), 10)
    print(rs)
    






