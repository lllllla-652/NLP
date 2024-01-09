# -*- encoding: utf-8 -*-
'''
@File    :   model_utils.py    
@Contact :   littlefish@88.com
@Blog    :   https://www.ilfishs.com

@Modify Time      @Author       @Version     @Desciption
------------      -------       --------     -----------
 2023/3/23 13:25  littlefish      1.0         None
 
 '''

import paddle
import paddle.nn as nn
from paddle.nn import Conv2D, Linear, Embedding


class MultiHeadSelfAttention(nn.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value):
        score = paddle.matmul(query, key, transpose_y=True)
        dim_key = paddle.cast(paddle.shape(key)[-1], 'float32')
        scaled_score = score / paddle.sqrt(dim_key)
        weights = nn.functional.softmax(scaled_score, axis=-1)
        output = paddle.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = paddle.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return paddle.transpose(x, perm=[0, 2, 1, 3])

    def forward(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = paddle.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = paddle.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = paddle.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
class PointWiseFeedForwardNetwork(nn.Layer):
    def __init__(self, embed_dim, feed_dim):
        super(PointWiseFeedForwardNetwork, self).__init__()
        # self.linear1 = paddle.fluid.dygraph.Linear(embed_dim, feed_dim, act='relu')
        self.linear1 = nn.Linear(embed_dim, feed_dim)
        self.softmax = nn.Softmax()
        self.linear2 = nn.Linear(feed_dim, embed_dim)

    def forward(self, x):
        out = self.softmax(self.linear1(x))
        out = self.linear2(out)
        return out
class TokenAndPositionEmbedding(nn.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        maxlen = paddle.shape(x)[-1]
        positions = paddle.arange(start=0, end=maxlen, step=1, dtype='int64')
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
class TransformerBlock(nn.Layer):
    def __init__(self, embed_dim, num_heads, feed_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(embed_dim, feed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim, epsilon=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, epsilon=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
class Transformer(nn.Layer):
    def __init__(self,maxlen,vocab_size,embed_dim,num_heads,feed_dim):
        super(Transformer, self).__init__()
        self.emb = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.trs = TransformerBlock(embed_dim, num_heads, feed_dim)
        self.drop1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(feed_dim, 20)
        self.relu = nn.ReLU()
        # self.relu = paddle.fluid.dygraph.Linear(feed_dim, 20, act='relu')
        self.drop2 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(20, 2)
        self.softmax = nn.Softmax()
        # self.soft = paddle.fluid.dygraph.Linear(20, 2, act='softmax')

    def forward(self, x):
        x = self.emb(x)
        x = self.trs(x)
        x = paddle.mean(x, axis=1)
        x = self.drop1(x)
        x = self.relu(self.linear1(x))
        x = self.drop2(x)
        x = self.softmax(self.linear2(x))
        return x

class LSTM(nn.Layer):
    def __init__(self,dict_dim):
        super(LSTM, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.class_dim = 2
        self.embedding = Embedding(
            self.dict_dim + 1, self.emb_dim,
            sparse=False)
        self._fc1 = Linear(self.emb_dim, self.hid_dim)
        self.lstm = paddle.nn.LSTM(self.hid_dim, self.hid_dim)
        self.fc2 = Linear(19200, self.class_dim)

    def forward(self, inputs):
        # [32, 150]
        emb = self.embedding(inputs)
        # [32, 150, 128]
        fc_1 = self._fc1(emb)
        # [32, 150, 128]
        x = self.lstm(fc_1)
        x = paddle.reshape(x[0], [0, -1])
        x = self.fc2(x)
        x = paddle.nn.functional.softmax(x)
        return x

class GRU(nn.Layer):
    def __init__(self,vocab_size,emb_size):
        super(GRU, self).__init__()

        # 将词汇表中的每个词映射为256维向量，以避免one-hot编码带来的稀疏性
        self.embedding = nn.Embedding(vocab_size, emb_size)

        # 定义模型
        # 这里的input_size就是词向量的维度，hidden_size就是RNN隐藏层的维度
        # 并不需要指定时间步数，也即seq_len，这是因为，GRU和LSTM都实现了自身的迭代
        self.gru = nn.GRU(input_size=emb_size, hidden_size=256, num_layers=2, direction='bidirectional', dropout=0.5)
        self.linear = nn.Linear(in_features=256 * 2, out_features=2)  # 二分类问题，情感分为积极和消极
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        # 词嵌入
        # inputs大小为句子长度*embedding向量大小=200*256
        # emb大小为 [32, 200, 256]
        emb = self.dropout(self.embedding(inputs))

        # 调用gru
        output, hidden = self.gru(emb)
        # output形状大小为[32, 200, 512]=[batch_size,seq_len,num_directions * hidden_size]
        # hidden形状大小为[4, 32, 256]=[num_layers * num_directions, batch_size, hidden_size]

        # 把前向的hidden与后向的hidden（大小为[32, 256]）合并在一起，对axis=1进行运算
        hidden = paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        # 合并后，hidden形状大小为[32, 512]=[batch_size, hidden_size * num_directions]

        hidden = self.dropout(hidden)
        return self.linear(hidden)

class CNN(nn.Layer):
    def __init__(self,dict_dim):
        super(CNN, self).__init__()
        self.dict_dim = dict_dim
        self.emb_dim = 128
        self.hid_dim = 128
        self.fc_hid_dim = 96
        self.class_dim = 2
        self.channels = 1
        self.win_size = [3, self.hid_dim]
        self.batch_size = 32
        self.seq_len = 150
        self.embedding = Embedding(self.dict_dim + 1, self.emb_dim, sparse=False)
        self.hidden1 = paddle.nn.Conv2D(in_channels=1,  # 通道数
                                        out_channels=self.hid_dim,  # 卷积核个数
                                        kernel_size=self.win_size,  # 卷积核大小
                                        padding=[1, 1]
                                        )
        self.relu1 = paddle.nn.ReLU()
        self.hidden3 = paddle.nn.MaxPool2D(kernel_size=2,  # 池化核大小
                                           stride=2)  # 池化步长2
        self.hidden4 = paddle.nn.Linear(128 * 75, 2)

    # 网络的前向计算过程
    def forward(self, input):
        # print('输入维度：', input.shape)
        x = self.embedding(input)
        x = paddle.reshape(x, [32, 1, 150, 128])
        x = self.hidden1(x)
        x = self.relu1(x)
        # print('第一层卷积输出维度：', x.shape)
        x = self.hidden3(x)
        # print('池化后输出维度：', x.shape)
        # 在输入全连接层时，需将特征图拉平会自动将数据拉平.

        x = paddle.reshape(x, shape=[self.batch_size, -1])
        out = self.hidden4(x)
        return out