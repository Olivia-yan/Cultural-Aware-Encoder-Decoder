import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import *




class AttentionAggregator(nn.Module):
    def __init__(self, h_dim):
        super(AttentionAggregator, self).__init__()
        self.linear = nn.Linear(h_dim, h_dim)
        self.context = nn.Parameter(torch.randn(h_dim))
        
    def forward(self, x):
        scores = torch.matmul(torch.tanh(self.linear(x)), self.context)
        weights = F.softmax(scores, dim=0).unsqueeze(1)   
        weighted_sum = torch.sum(weights * x, dim=0, keepdim=True)  
        return weighted_sum


class CrossRelationFusion(nn.Module):
    def __init__(self, num_rels, h_dim, num_heads=4, dropout=0.1):
        super(CrossRelationFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=num_heads, dropout=dropout)
        self.linear = nn.Linear(h_dim, h_dim)
        
    def forward(self, rel_embs):
        rel_embs = rel_embs.unsqueeze(1) 
        attn_output, _ = self.multihead_attn(rel_embs, rel_embs, rel_embs)
        attn_output = self.linear(attn_output)
        return attn_output.squeeze(1) 


class GatedFusion(nn.Module):
    def __init__(self, h_dim):
        super(GatedFusion, self).__init__()
        self.linear = nn.Linear(2 * h_dim, h_dim)
    
    def forward(self, current_h, prev_h):
        combined = torch.cat([current_h, prev_h], dim=1) 
        gate = torch.sigmoid(self.linear(combined))         
        fused = gate * current_h + (1 - gate) * prev_h       
        return fused

class TemporalDifferenceFusion(nn.Module):
    def __init__(self, h_dim):
        super(TemporalDifferenceFusion, self).__init__()
        self.linear = nn.Linear(h_dim, h_dim)
    
    def forward(self, current_h, prev_h):
        delta = current_h - prev_h   
        delta_processed = F.relu(self.linear(delta))  
        fused = current_h + delta_processed
        return fused



class SpatioTemporalFusion(nn.Module):
    def __init__(self, h_dim, num_heads=4):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(h_dim, num_heads)
        self.spatial_conv = nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1)
        self.gate = nn.Sequential(
            nn.Linear(2*h_dim, h_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, hist_embs):
        temp_out, _ = self.temporal_attn(
            x.unsqueeze(0), 
            torch.stack(hist_embs[-5:]).transpose(0,1),
            torch.stack(hist_embs[-5:]).transpose(0,1)
        )
        spatial_out = self.spatial_conv(x.unsqueeze(-1)).squeeze()
        gate = self.gate(torch.cat([temp_out.squeeze(0), spatial_out], dim=1))
        return gate * temp_out + (1 - gate) * spatial_out

class MemoryEnhancedRelationCell(nn.Module):
    def __init__(self, h_dim, memory_size=64):
        super().__init__()
        self.memory = nn.Parameter(torch.Tensor(memory_size, h_dim))
        nn.init.xavier_uniform_(self.memory)
        self.memory_attn = nn.MultiheadAttention(h_dim, num_heads=4)
        self.gru_cell = nn.GRUCell(2*h_dim, h_dim)

    def forward(self, current_rel, history_rels):
        mem_out, _ = self.memory_attn(
            current_rel.unsqueeze(0),
            torch.cat([self.memory, history_rels[-3:]]).unsqueeze(0),
            torch.cat([self.memory, history_rels[-3:]]).unsqueeze(0)
        )
        fused = torch.cat([current_rel, mem_out.squeeze(0)], dim=1)
        return self.gru_cell(fused, history_rels[-1])


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.triplet = nn.TripletMarginLoss(margin=1.0)
        self.alpha = alpha

    def forward(self, pred, target, anchor, pos, neg):
        return self.alpha*self.ce(pred, target) + \
               (1-self.alpha)*self.triplet(anchor, pos, neg)





class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times, time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None
        self.filter_convs = nn.Conv1d(in_channels=self.num_rels*2,out_channels=self.num_rels*2,kernel_size=1,dilation=1)
        self.cross_relation_proj = nn.Linear(self.h_dim, self.h_dim)
        self.relation_queries = nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim))
        self.time_attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=4)
        self.time_norm = nn.LayerNorm(h_dim) if layer_norm else None
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(h_dim, h_dim, kernel_size=k) 
            for k in [3, 5, 7]  
            ])
        self.temporal_fuse = nn.Linear(3*h_dim, h_dim)
        self.att_agg = AttentionAggregator(self.h_dim)
        self.cross_relation_fusion = CrossRelationFusion(num_rels=self.num_rels*2, h_dim=self.h_dim, num_heads=4, dropout=0.1)
        self.gated_fusion = GatedFusion(self.h_dim)
        self.temp_diff_fusion = TemporalDifferenceFusion(self.h_dim)


        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))


        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)


        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight , gain=nn.init.calculate_gain('relu'))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)


        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)


        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

    def transform_tensor(self, input_tensor):
        transformed_first_dim = input_tensor[:460]

        transformed_second_dim = torch.nn.functional.interpolate(
            transformed_first_dim.permute(2, 3, 0, 1),  
            size=(460, 200),
            mode='bilinear',
            align_corners=False
        ).permute(2, 3, 0, 1) 
        

        transformed_tensor = transformed_second_dim
        
        return transformed_tensor

    def transform_11(self, input_tensor):
        
        transformed_tensor = torch.nn.functional.interpolate(
            input_tensor,
            size=(460, 400),
            mode='bilinear',
            align_corners=False
        )
        
        return transformed_tensor

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []
        history_rel_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
        
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]

                shared_features = self.cross_relation_proj(x)  
                
                

                query = self.relation_queries[r_idx].unsqueeze(0)
                attn_scores = torch.matmul(shared_features, query.t()) / (self.h_dim ** 0.5)
                attn_weights = F.softmax(attn_scores, dim=0)
                
                relation_embed = torch.sum(x * attn_weights, dim=0)
                x_input[r_idx] = relation_embed
                if x.shape[0] == 0:
                    continue
                x_weighted = self.att_agg(x)  
                x_input[r_idx] = x_weighted.squeeze(0)
           
            x_input = self.cross_relation_fusion(x_input)
            


           
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)  
                reshaped_tensor = x_input.unsqueeze(2).unsqueeze(2)
                kernel_size=reshaped_tensor.shape[2]
                block = CBAMBlock(channel=240,reduction=20,kernel_size=kernel_size).to(self.gpu)
                reshaped_tensor = block(reshaped_tensor)   
                x_input = reshaped_tensor.squeeze(2).squeeze(2)            
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                history_rel_embs = self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                reshaped_tensor = x_input.unsqueeze(2).unsqueeze(2)
                kernel_size=reshaped_tensor.shape[2]
                block = CBAMBlock(channel=240,reduction=20,kernel_size=kernel_size).to(self.gpu)
                reshaped_tensor = block(reshaped_tensor)   
                x_input = reshaped_tensor.squeeze(2).squeeze(2)
                
                prev_rel = history_rel_embs[-1]  
                prev_rel_2d = prev_rel.unsqueeze(1)
                
                compose_weights = torch.matmul(self.h_0, prev_rel_2d)  
                compose_weights = compose_weights.squeeze(1)
                compose_weights = F.softmax(compose_weights, dim=0)
                
                compose_weights_2d = compose_weights.unsqueeze(1)
                composed_feat = torch.matmul(compose_weights_2d, prev_rel_2d.T)  

                self.h_0 = self.h_0 + composed_feat
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                history_rel_embs = self.h_0


 

           
            reshaped_tensor = self.h_0.unsqueeze(2).unsqueeze(3)   
            para = ParallelPolarizedSelfAttention(channel=120).to(self.gpu)
            reshaped_tensor=para(reshaped_tensor)
            self.h_0 = reshaped_tensor.squeeze(2).squeeze(2)



            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])       
            f = torch.tanh(current_h)
            g = torch.sigmoid(current_h)
            current_h = f*g
            current_h = F.normalize(current_h) if self.layer_norm else current_h

            fused_h = self.gated_fusion(current_h, self.h)


            reshaped_ch = current_h.unsqueeze(2).unsqueeze(2)
            kernel_size=reshaped_ch.shape[2]
            cbam = CBAMBlock(channel=120,reduction=20,kernel_size=kernel_size).to(self.gpu)
            reshaped_ch=cbam(reshaped_ch)
            current_h = reshaped_ch.squeeze(2).squeeze(2)


            
            self.h = self.entity_cell_1(current_h, self.h)


            reshaped_tensor = self.h.unsqueeze(2).unsqueeze(2)        
            block = Partial_conv3(120, 2, 'split_cat').cuda()
            reshaped_tensor = block(reshaped_tensor)
            self.h = reshaped_tensor.squeeze(2).squeeze(2)
        

            self.h = F.normalize(self.h) if self.layer_norm else self.h


            if i > 0:  

                hist_stack = torch.stack(history_embs, dim=0)

                attn_out, _ = self.time_attn(
                    query=self.h.unsqueeze(0),  
                    key=hist_stack,
                    value=hist_stack
                )
                attn_out = attn_out.squeeze(0)

                self.h = self.h + attn_out
                if self.layer_norm:
                    self.h = self.time_norm(self.h)




            history_embs.append(self.h)
            


        
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)

            score_rel_r = self.rel_raw_mode(embedding, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples)
            score_h = self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)

            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.log(score_rel)
            score = self.history_rate * score_h + (1 - self.history_rate) * score_r
            score = torch.log(score)

            return all_triples, score, score_rel


    def get_loss(self, glist, triples, static_graph, entity_history_vocabulary, rel_history_vocabulary, use_cuda):
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        time_embs = self.get_init_time(all_triples)


        if self.entity_prediction:
            score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_h = self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
            score_en = self.history_rate * score_h + (1 - self.history_rate) * score_r
            scores_en = torch.log(score_en)
            loss_ent += F.nll_loss(scores_en, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, time_embs, all_triples, rel_history_vocabulary)
            score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            scores_re = torch.log(score_re)
            loss_rel += F.nll_loss(scores_re, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding = global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        score = F.softmax(scores_re, dim=1)
        return score

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h
