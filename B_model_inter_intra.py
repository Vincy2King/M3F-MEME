import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
# DEVICE = torch.cuda.set_device(2)
DEVICE = torch.device("cuda:1")
mode = 'dijin'
is_split = True

class GatingLayer(nn.Module):
    def __init__(self, input_size):
        super(GatingLayer, self).__init__()

        # Define the linear layer for the gating network
        self.gating_network = nn.Linear(input_size, input_size)

    def forward(self, x):
        # Compute the gating scores
        gating_scores = torch.sigmoid(self.gating_network(x))

        # Multiply the input by the gating scores to get the gated output
        gated_output = x * gating_scores

        return gated_output


class Co_attention(nn.Module):
    def __init__(self, hidden_size):
        super(Co_attention, self).__init__()
        self.hidden_size = hidden_size
        self.text_linear = nn.Linear(hidden_size, hidden_size)
        self.meta_linear = nn.Linear(hidden_size, hidden_size)
        self.concat_linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, text_emb, meta_emb):
        # Compute affinity matrix
        affinity = torch.matmul(self.text_linear(text_emb), self.meta_linear(meta_emb).transpose(0, 1))
        # affinity = torch.softmax(affinity, dim=1)

        text_att = torch.softmax(affinity, dim=1)
        meta_att = torch.softmax(affinity.transpose(0, 1), dim=1)

        # Compute context vectors
        meta_ctx = torch.matmul(meta_att, meta_emb)
        text_ctx = torch.matmul(text_att, text_emb)

        # Concatenate and apply linear layer
        out = torch.cat([text_ctx, meta_ctx], dim=1)
        # print('out:',out.size())
        out = self.concat_linear(out)
        # out = text_ctx+meta_ctx

        return out

class CrossAttentionTransformer(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttentionTransformer, self).__init__()

        # Multi-head attention layers for text and image features
        self.attention_text = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)#.to(device)
        self.attention_image = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)#.to(device)

        # Feedforward layers for text and image features
        self.feedforward_text = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim , input_dim)
        )#.to(device)
        self.feedforward_image = nn.Sequential(
            nn.Linear(input_dim, input_dim ),
            nn.ReLU(),
            nn.Linear(input_dim , input_dim)
        )#.to(device)

    def forward(self, text_features, image_features):
        # print(text_features.size(),image_features.size())
        # Compute multi-head attention for text features
        attn_output_text, attn_output_weights_text = self.attention_text(query=text_features, key=image_features, value=image_features)
        # Compute multi-head attention for image features
        attn_output_image, attn_output_weights_image = self.attention_image(query=image_features, key=text_features, value=text_features)

        # Apply feedforward layers to the multi-head attention outputs
        attn_output_text = self.feedforward_text(attn_output_text)
        attn_output_image = self.feedforward_image(attn_output_image)

        return attn_output_text, attn_output_image


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text, meta):
        query = self.query_layer(text)
        key = self.key_layer(meta)
        value = self.value_layer(meta)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.hidden_size).float())

        attn_weights = self.softmax(scores)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output

class CrossAttention(nn.Module):
    def __init__(self, text_dim, img_dim):
        super(CrossAttention, self).__init__()
        self.text_dim = text_dim
        self.img_dim = img_dim

        self.Wt = nn.Linear(text_dim, img_dim)
        self.Wi = nn.Linear(img_dim, text_dim)

    def forward(self, text_features, img_features):
        # project text features to image feature space
        text_features_proj = self.Wt(text_features)  # shape: [batch_size, img_dim]

        # compute image-text similarity scores
        scores = torch.mm(img_features, text_features_proj.t())  # shape: [batch_size, batch_size]

        # compute image attention weights
        img_att_weights = F.softmax(scores, dim=0)  # shape: [batch_size, batch_size]

        # compute image attended text features
        text_features_attended = torch.mm(img_att_weights, self.Wi(img_features))  # shape: [batch_size, text_dim]

        # compute text attention weights
        text_att_weights = F.softmax(scores.t(), dim=0)  # shape: [text_dim, img_dim]

        # compute text attended image features
        img_features_attended = torch.mm(text_att_weights, self.Wt(text_features))  # shape: [batch_size, img_dim]

        return text_features_attended, img_features_attended

class GatingLayer1(nn.Module):
    def __init__(self, hidden_size):
        super(GatingLayer1, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size).to(DEVICE)
        self.key = nn.Linear(hidden_size, hidden_size).to(DEVICE)
        self.value = nn.Linear(hidden_size, hidden_size).to(DEVICE)
        self.softmax = nn.Softmax(dim=1).to(DEVICE)

    def forward(self, x):
        # x: (seq_len, batch_size, hidden_size)

        # Compute query, key, value
        Q = self.query(x)#.to(DEVICE)  # (seq_len, batch_size, hidden_size)
        K = self.key(x)  # (seq_len, batch_size, hidden_size)
        V = self.value(x)  # (seq_len, batch_size, hidden_size)

        # Compute scores
        scores = torch.matmul(Q.transpose(0, 1), K.transpose(0, 1).transpose(-1, -2))  # (batch_size, seq_len, seq_len)
        scores = scores / (K.shape[-1] ** 0.5)  # Scale by square root of key dimension

        # Compute attention weights
        attn_weights = self.softmax(scores)  # (batch_size, seq_len, seq_len)

        # Compute weighted sum of values
        context = torch.matmul(attn_weights, V.transpose(0, 1))  # (batch_size, seq_len, hidden_size)
        context = context.transpose(0, 1)  # (seq_len, batch_size, hidden_size)

        # Compute gated output
        output = x.mean(dim=0) + context.mean(dim=0)  # (batch_size, hidden_size)

        return output


class mergeNet(nn.Module):
    def __init__(self,device,class_sentiment,class_intention,class_offensiveness,
                 class_m_occurrence,class_m_category,
                 is_mld,is_inter,is_intra,cat_or_add,feature_mode,share_or_single):
        super(mergeNet, self).__init__()

        after_text = 768
        after_img = 768
        linear_output = 128

        self.is_mld = is_mld
        self.is_inter = is_inter
        self.is_intra = is_intra
        self.cat_or_add = cat_or_add

        if feature_mode =='resnet':
            print('resnet!!!')
            self.bn_input2 = nn.BatchNorm1d(2048, momentum=0.1).to(device)
            self.fc = nn.Linear(2048, 768).to(device)
            self.fc_img_dim = nn.Linear(2048, after_img).to(device)
            dimension = 2048
        else:
            dimension = 4096

        self.bn_input1 = nn.BatchNorm1d(768, momentum=0.1).to(device)  # 首先定义了一个归一化的函数，需要归一化的维度为768；num_features：一般输入参数为NCH*W，即为其中特征的数量；momentum – 移动平均的动量值
        self.bn_input2 = nn.BatchNorm1d(dimension, momentum=0.1).to(device)
        self.fc_text_dim = nn.Linear(768, after_text).to(device)
        self.fc_img_dim = nn.Linear(dimension, after_img).to(device)

        self.fc = nn.Linear(dimension, 768).to(device)


        self.ln_input1 = nn.LayerNorm(512, eps=1e-05, elementwise_affine=True).to(device)

        self.fc1 = nn.Linear(768, 512).to(device)

        # self.fc2_sentiment = nn.Linear(512, class_sentiment).to(device)
        # self.fc2_sentiment = nn.Linear(512, class_sentiment).to(device)
        # self.fc2_sentiment = nn.Linear(512, class_sentiment).to(device)
        # self.fc2_sentiment = nn.Linear(512, class_sentiment).to(device)
        # self.fc2_sentiment = nn.Linear(512, class_sentiment).to(device)

        self.cross_attention = CrossAttentionTransformer(input_dim=768).to(device) #CrossAttention(text_dim=768, img_dim=4096)



        # self.intra_attention = Attention(after_text).to(device)
        self.intra_attention = Co_attention(after_text).to(device)
        self.concat_intra = nn.Linear(after_text * 2, after_text).to(device)

        self.all_fc1_1 = nn.Linear(768*3, class_sentiment).to(device)
        self.all_fc1_2 = nn.Linear(768*3, class_intention).to(device)
        self.all_fc1_3 = nn.Linear(768*3, class_offensiveness).to(device)
        self.all_fc1_4 = nn.Linear(768*3, class_m_occurrence).to(device)
        self.all_fc1_5 = nn.Linear(768*3, class_m_occurrence).to(device)

        self.all_fc2_1 = nn.Linear(512, class_sentiment ).to(device)
        self.all_fc2_2 = nn.Linear(512, class_intention).to(device)
        self.all_fc2_3 = nn.Linear(512, class_offensiveness).to(device)
        self.all_fc2_4 = nn.Linear(512, class_m_occurrence).to(device)
        self.all_fc2_5 = nn.Linear(512, class_m_occurrence).to(device)

        self.all_fc_5 = nn.Linear(768+dimension, class_m_occurrence).to(device)

        # Linear layer
        # self.Linear_four =  nn.Linear(after_text*2 + after_img, linear_output).to(device)
        self.Linear_four =  nn.Linear(after_text + after_img , linear_output).to(device)
        self.Linear_SIOM = nn.Linear(after_text + after_img, linear_output).to(device)

        self.Linear_three = nn.Linear(after_text + after_img , linear_output).to(device)
        self.Linear_SIO = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_SIM = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_IOM = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_SOM = nn.Linear(after_text + after_img, linear_output).to(device)

        self.Linear_two = nn.Linear(after_text + after_img , linear_output).to(device)
        self.Linear_SI = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_SO = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_SM = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_IO = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_IM = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_OM = nn.Linear(after_text + after_img, linear_output).to(device)

        self.Linear_one = nn.Linear(after_text + after_img , linear_output).to(device)
        self.Linear_S = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_I = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_O = nn.Linear(after_text + after_img, linear_output).to(device)
        self.Linear_M = nn.Linear(after_text + after_img, linear_output).to(device)
        number = 0
        if mode == 'include':
            self.gating_layer_1 = GatingLayer1(hidden_size=linear_output*15).to(DEVICE)
            self.gating_layer_2 = GatingLayer1(hidden_size=linear_output*7).to(DEVICE)
            self.gating_layer_3 = GatingLayer1(hidden_size=linear_output*3).to(DEVICE)
            self.gating_layer_4 = GatingLayer1(hidden_size=linear_output*8).to(DEVICE)
            number = 15 * 1 + 7 * 3 + 3 * 3 + 8
        elif mode == 'dijin':
            self.gating_layer_1 = GatingLayer1(hidden_size=linear_output * 5).to(DEVICE)
            self.gating_layer_2 = GatingLayer1(hidden_size=linear_output * 4).to(DEVICE)
            self.gating_layer_3 = GatingLayer1(hidden_size=linear_output * 3).to(DEVICE)
            self.gating_layer_4 = GatingLayer1(hidden_size=linear_output * 1).to(DEVICE)
            number = 5 * 1 + 4 * 3 + 3 * 3 + 1
        elif mode == 'woself':
            self.gating_layer_1 = GatingLayer1(hidden_size=linear_output * 4).to(DEVICE)
            self.gating_layer_2 = GatingLayer1(hidden_size=linear_output * 3).to(DEVICE)
            self.gating_layer_3 = GatingLayer1(hidden_size=linear_output * 2).to(DEVICE)
            self.gating_layer_4 = GatingLayer1(hidden_size=linear_output * 1).to(DEVICE)
            number = 4 * 1 + 3 * 3 + 2 * 3 + 1

        self.Linear_1 = nn.Linear(linear_output * 5, linear_output).to(device)
        self.Linear_2 = nn.Linear(linear_output * 4, linear_output).to(device)
        self.Linear_3 = nn.Linear(linear_output * 3, linear_output).to(device)
        self.Linear_4 = nn.Linear(linear_output * 1, linear_output).to(device)

        self.gating_layer_sentiment = GatingLayer1(hidden_size=linear_output * number).to(DEVICE)
        self.gating_layer_intention = GatingLayer1(hidden_size=linear_output * number).to(DEVICE)
        self.gating_layer_offensiveness = GatingLayer1(hidden_size=linear_output * number).to(DEVICE)
        self.gating_layer_m_occurrence = GatingLayer1(hidden_size=linear_output * number).to(DEVICE)

        self.gating_layer_sentiment2 = GatingLayer1(hidden_size=linear_output ).to(DEVICE)
        self.gating_layer_intention2 = GatingLayer1(hidden_size=linear_output ).to(DEVICE)
        self.gating_layer_offensiveness2 = GatingLayer1(hidden_size=linear_output ).to(DEVICE)
        self.gating_layer_m_occurrence2 = GatingLayer1(hidden_size=linear_output).to(DEVICE)

        self.classifier_sentiment = nn.Linear(linear_output * number, class_sentiment).to(device)
        self.classifier_intention = nn.Linear(linear_output * number, class_intention).to(device)
        self.classifier_offensiveness = nn.Linear(linear_output * number, class_offensiveness).to(device)
        self.classifier_m_occurrence = nn.Linear(linear_output * number, class_m_occurrence).to(device)
        self.classifier_m_category = nn.Linear(linear_output * number, class_m_category).to(device)

        self.classifier_sentiment2 = nn.Linear(linear_output , class_sentiment).to(device)
        self.classifier_intention2 = nn.Linear(linear_output, class_intention).to(device)
        self.classifier_offensiveness2 = nn.Linear(linear_output , class_offensiveness).to(device)
        self.classifier_m_occurrence2 = nn.Linear(linear_output, class_m_occurrence).to(device)
        self.classifier_m_category2 = nn.Linear(linear_output, class_m_category).to(device)

        self.softmax_sentiment = nn.Linear(after_text * 2, class_sentiment).to(device)
        self.softmax_intention = nn.Linear(after_text * 2, class_intention).to(device)
        self.softmax_offensiveness = nn.Linear(after_text * 2, class_offensiveness).to(device)
        self.softmax_m_occurrence = nn.Linear(after_text * 2, class_m_occurrence).to(device)
        self.softmax_m_category = nn.Linear(after_text * 2, class_m_category).to(device)

        self.two_to_one = nn.Linear(after_text*2, after_text).to(device)

    def forward(self, text, meta,meta_pad, img,source,target,text_mate):
        text = self.bn_input1(text)
        source = self.bn_input1(source)
        target = self.bn_input1(target)
        text_mate = self.bn_input1(text_mate)
        img = self.bn_input2(img)
        meta_hidden = torch.cat([text, img], dim=1)
        out5 = F.softmax(self.all_fc_5(meta_hidden), dim=1)
        values, indice = out5.cpu().max(1)
        # print('indice:', indice)

        # print('meta_pad:',meta_pad)
        # print(meta.cpu().float().detach().numpy())
        for k in range(len(indice)):
            if indice[k] == 0:
                # print('before:',meta[k],meta_pad[k])
                source[k] = meta_pad[k]
                target[k] =meta_pad[k]
        align_text = self.fc_text_dim(text)
        align_img = self.fc_img_dim(img)

        meta_text = self.two_to_one(torch.cat([source,target],dim=1))
        intra_emb = self.intra_attention(text, meta_text)

        text_features_attended, img_features_attended = self.cross_attention(align_text.unsqueeze(0),
                                                                             align_img.unsqueeze(0))
        text_features_attended, img_features_attended = text_features_attended.squeeze(), img_features_attended.squeeze()
        final_emb = torch.cat([text_features_attended, img_features_attended], dim=1)

        # exit()
        if self.cat_or_add == 'cat':
            final_emb = torch.cat([final_emb,intra_emb],dim=1)
            # print(final_emb.size())
            out1 = F.softmax(self.all_fc1_1(final_emb), dim=1)  # 32*2
            out2 = F.softmax(self.all_fc1_2(final_emb), dim=1)  # 32*2
            out3 = F.softmax(self.all_fc1_3(final_emb), dim=1)  # 32*2
            out4 = F.softmax(self.all_fc1_4(final_emb), dim=1)  # 32*2
            # out5 = F.softmax(self.all_fc1_5(final_emb), dim=1)  # 32*2
        else:
            img = F.relu(self.fc(img)) # 768
            cat_vec = torch.add(text_mate, img) / 2
            cat_vec = F.dropout(F.relu(self.fc1(cat_vec)), 0.4)
            cat_vec = self.ln_input1(cat_vec)  # 32*512
            out1 = F.softmax(self.all_fc2_1(cat_vec), dim=1)  # 32*2
            out2 = F.softmax(self.all_fc2_2(cat_vec), dim=1)  # 32*2
            out3 = F.softmax(self.all_fc2_3(cat_vec), dim=1)  # 32*2
            out4 = F.softmax(self.all_fc2_4(cat_vec), dim=1)  # 32*2
            # out5 = F.softmax(self.all_fc2_5(cat_vec), dim=1)  # 32*2


        # print('meta_hidden:',meta_hidden.size())


        return out1, out2, out3, out4, out5