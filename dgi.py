import torch
import torch.nn as nn

class DGI(nn.Module):
    def __init__(self, n_h):
        super(DGI, self).__init__()
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2,t, msk=None, samp_bias1=None, samp_bias2=None):
        seq1 = seq1.unsqueeze(0)
        seq2 = seq2.unsqueeze(0)
        t = t.unsqueeze(0)
        c = self.read(t, msk)
        c = self.sigm(c)
        ret = self.disc(c, seq1, seq2, samp_bias1, samp_bias2)#seq1是负样本，seq2是正样本

        return ret


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)## 计算h_pl与c_x的相似度得分，h_pl是负样本
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)## 计算h_mi与c_x的相似度得分，h_mi是正样本

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)




class MI(nn.Module):
    def __init__(self, feature_dim):
        super(MI, self).__init__()
        # 双线性变换层，用于计算互信息得分
        self.bilinear = nn.Bilinear(feature_dim, feature_dim, 1)

    def forward(self, feat1, feat2):
        """
        计算 feat1 和 feat2 之间的互信息得分
        :param feat1: 特征1 [batch_size, feature_dim]
        :param feat2: 特征2 [batch_size, feature_dim]
        :return: 互信息得分 [batch_size, 1]
        """
        # 使用双线性变换计算相似度得分
        mi_score = self.bilinear(feat1, feat2)
        return mi_score


class Discriminator_MI(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_MI, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c), 2)## 计算h_pl与c的相似度得分，h_pl是负样本
        sc_2 = torch.squeeze(self.f_k(h_mi, c), 2)## 计算h_mi与c的相似度得分，h_mi是正样本

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits




