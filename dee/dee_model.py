# Code Reference: (https://github.com/dolphin-zs/Doc2EDAG)

import torch
from torch import nn
import torch.nn.functional as F
import math
from collections import OrderedDict, namedtuple, defaultdict
import random

from . import transformer
from .ner_model import NERModel
import dgl
import dgl.nn.pytorch as dglnn

DocSpanInfo = namedtuple(
    'DocSpanInfo', (
        'span_token_tup_list',  # [(span_token_id, ...), ...], num_spans
        'span_dranges_list',  # [[(sent_idx, char_s, char_e), ...], ...], num_spans
        'span_mention_range_list',  # [(mention_idx_s, mention_idx_e), ...], num_spans
        'mention_drange_list',  # [(sent_idx, char_s, char_e), ...], num_mentions
        'mention_type_list',  # [mention_type_id, ...], num_mentions
        'event_dag_info',  # event_idx -> field_idx -> pre_path -> cur_span_idx_set
        'missed_sent_idx_list',  # index list of sentences where gold spans are not extracted
    )
)

def get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=False):
    assert len(doc_token_types_list) == len(doc_fea_list)
    doc_span_info_list = []
    for doc_token_types, doc_fea in zip(doc_token_types_list, doc_fea_list):
        doc_token_type_mat = doc_token_types.tolist()  # [[token_type, ...], ...]

        # using extracted results is also ok
        # span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)
        if use_gold_span:
            span_token_tup_list = doc_fea.span_token_ids_list
            span_dranges_list = doc_fea.span_dranges_list
        else:
            span_token_tup_list, span_dranges_list = extract_doc_valid_span_info(doc_token_type_mat, doc_fea)

            if len(span_token_tup_list) == 0:
                # do not get valid entity span results,
                # just use gold spans to avoid crashing at earlier iterations
                # TODO: consider generate random negative spans
                span_token_tup_list = doc_fea.span_token_ids_list
                span_dranges_list = doc_fea.span_dranges_list

        # one span may have multiple mentions
        span_mention_range_list, mention_drange_list, mention_type_list = get_span_mention_info(
            span_dranges_list, doc_token_type_mat
        )

        # generate event decoding dag graph for model training
        event_dag_info, _, missed_sent_idx_list = doc_fea.generate_dag_info_for(span_token_tup_list, return_miss=True)

        # doc_span_info will incorporate all span-level information needed for the event extraction
        doc_span_info = DocSpanInfo(
            span_token_tup_list, span_dranges_list, span_mention_range_list,
            mention_drange_list, mention_type_list,
            event_dag_info, missed_sent_idx_list,
        )

        doc_span_info_list.append(doc_span_info)

    return doc_span_info_list

class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)
        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}

class GITModel(nn.Module):
    """Document-level Event Extraction Model"""

    def __init__(self, config, event_type_fields_pairs, ner_model=None):
        super(GITModel, self).__init__()
        # Note that for distributed training, you must ensure that
        # for any batch, all parameters need to be used

        self.config = config
        self.event_type_fields_pairs = event_type_fields_pairs

        if ner_model is None:
            self.ner_model = NERModel(config)
        else:
            self.ner_model = ner_model

        # all event tables
        self.event_tables = nn.ModuleList([
            EventTable(event_type, field_types, config.hidden_size)
            for event_type, field_types in self.event_type_fields_pairs
        ])

        # sentence position indicator
        self.sent_pos_encoder = SentencePosEncoder(
            config.hidden_size, max_sent_num=config.max_sent_num, dropout=config.dropout
        )

        self.ment_type_encoder = MentionTypeEncoder(
            config.hidden_size, config.num_entity_labels, dropout=config.dropout
        )

        # various attentive reducer
        if self.config.seq_reduce_type == 'AWA':
            self.doc_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
            self.span_token_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
            self.span_mention_reducer = AttentiveReducer(config.hidden_size, dropout=config.dropout)
        else:
            assert self.config.seq_reduce_type in {'MaxPooling', 'MeanPooling'}

        if self.config.use_path_mem:
            # get field-specific and history-aware information for every span
            self.field_context_encoder = transformer.make_transformer_encoder(
                config.num_tf_layers, config.hidden_size, ff_size=config.ff_size, dropout=config.dropout
            )
        
        self.rel_name_lists = ['m-m', 's-m', 's-s']
        self.gcn_layers = config.gcn_layer
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(config.hidden_size, config.hidden_size, self.rel_name_lists,
                                num_bases=len(self.rel_name_lists), activation=nn.ReLU(), self_loop=True, dropout=config.dropout)
                                for i in range(self.gcn_layers)])
        self.middle_layer = nn.Sequential(
            nn.Linear(config.hidden_size * (config.gcn_layer+1), config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)    
        )

        self.sent_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.mention_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.intra_path_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.inter_path_embedding = nn.Parameter(torch.randn(config.hidden_size))
        
    def get_batch_sent_emb(self, ner_token_emb, ner_token_masks, valid_sent_num_list):
        # From [ner_batch_size, sent_len, hidden_size] to [ner_batch_size, hidden_size]
        if self.config.seq_reduce_type == 'AWA':
            total_sent_emb = self.doc_token_reducer(ner_token_emb, masks=ner_token_masks)
        elif self.config.seq_reduce_type == 'MaxPooling':
            total_sent_emb = ner_token_emb.max(dim=1)[0]
        elif self.config.seq_reduce_type == 'MeanPooling':
            total_sent_emb = ner_token_emb.mean(dim=1)
        else:
            raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

        total_sent_pos_ids = []
        for valid_sent_num in valid_sent_num_list:
            total_sent_pos_ids += list(range(valid_sent_num))
        total_sent_emb = self.sent_pos_encoder(total_sent_emb, sent_pos_ids=total_sent_pos_ids)

        return total_sent_emb

    def get_event_cls_info(self, sent_context_emb, doc_fea, train_flag=True):
        doc_event_logps = []
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            event_table = self.event_tables[event_idx]
            cur_event_logp = event_table(sent_context_emb=sent_context_emb)  # [1, hidden_size]
            doc_event_logps.append(cur_event_logp)
        doc_event_logps = torch.cat(doc_event_logps, dim=0)  # [num_event_types, 2]

        if train_flag:
            device = doc_event_logps.device
            doc_event_labels = torch.tensor(
                doc_fea.event_type_labels, device=device, dtype=torch.long, requires_grad=False
            )  # [num_event_types]
            doc_event_cls_loss = F.nll_loss(doc_event_logps, doc_event_labels, reduction='sum')
            return doc_event_cls_loss
        else:
            doc_event_pred_list = doc_event_logps.argmax(dim=-1).tolist()
            return doc_event_pred_list

    def get_field_cls_info(self, event_idx, field_idx, batch_span_emb,
                           batch_span_label=None, train_flag=True):
        batch_span_logp = self.get_field_pred_logp(event_idx, field_idx, batch_span_emb)

        if train_flag:
            assert batch_span_label is not None
            device = batch_span_logp.device
            data_type = batch_span_logp.dtype
            # to prevent too many FPs
            class_weight = torch.tensor(
                [self.config.neg_field_loss_scaling, 1.0], device=device, dtype=data_type, requires_grad=False
            )
            field_cls_loss = F.nll_loss(batch_span_logp, batch_span_label, weight=class_weight, reduction='sum')
            return field_cls_loss, batch_span_logp
        else:
            span_pred_list = batch_span_logp.argmax(dim=-1).tolist()
            return span_pred_list, batch_span_logp

    def get_field_pred_logp(self, event_idx, field_idx, batch_span_emb, include_prob=False):
        event_table = self.event_tables[event_idx]
        batch_span_logp = event_table(batch_span_emb=batch_span_emb, field_idx=field_idx)

        if include_prob:
            # used for decision sampling, is not inside the computation graph
            batch_span_prob = batch_span_logp.detach().exp()
            return batch_span_logp, batch_span_prob
        else:
            return batch_span_logp

    def conduct_field_level_reasoning(self, event_idx, field_idx, prev_decode_context, prev_global_path_memory, global_path_memory, batch_span_context, sent_num):
        event_table = self.event_tables[event_idx]
        field_query = event_table.field_queries[field_idx]
        num_spans = batch_span_context.size(0)
        # make the model to be aware of which field
        batch_cand_emb = batch_span_context + field_query
        new_prev_decode_context = prev_decode_context.clone()
        new_prev_decode_context[sent_num:] += self.intra_path_embedding
        if self.config.use_path_mem:
            if prev_global_path_memory is None:
                # [1, num_spans + valid_sent_num, hidden_size]
                total_cand_emb = torch.cat([batch_cand_emb, new_prev_decode_context, global_path_memory[0]+self.inter_path_embedding], dim=0).unsqueeze(0)
            else:
                total_cand_emb = torch.cat([batch_cand_emb, new_prev_decode_context, prev_global_path_memory, global_path_memory[0]+self.inter_path_embedding], dim=0).unsqueeze(0)
            # use transformer to do the reasoning
            total_cand_emb = self.field_context_encoder(total_cand_emb, None).squeeze(0)
            batch_cand_emb = total_cand_emb[:num_spans, :]
        # TODO: what if reasoning over reasoning context
        return batch_cand_emb, prev_decode_context

    def get_field_mle_loss_list(self, doc_sent_context, batch_span_context,
                                event_idx, field_idx2pre_path2cur_span_idx_set,
                                prev_global_path_memory=None):
        field_mle_loss_list = []
        num_fields = self.event_tables[event_idx].num_fields
        num_spans = batch_span_context.size(0)
        sent_num = doc_sent_context.size(0)
        prev_path2prev_decode_context = {
            (): doc_sent_context
        }
        prev_path2global_memory_idx = {
            (): 0
        }
        global_path_memory = self.event_tables[event_idx].rnn_cell(self.event_tables[event_idx].event_query)

        for field_idx in range(num_fields):
            prev_path2cur_span_idx_set = field_idx2pre_path2cur_span_idx_set[field_idx]
            span_context_bank = []
            prev_global_memory_idx_list = []
            
            for prev_path, cur_span_idx_set in prev_path2cur_span_idx_set.items():
                if prev_path not in prev_path2prev_decode_context:
                    # note that when None and valid_span co-exists, ignore None paths during training
                    continue
                # get decoding context
                prev_decode_context = prev_path2prev_decode_context[prev_path]
                # conduct reasoning on this field
                batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                    event_idx, field_idx, prev_decode_context, prev_global_path_memory, global_path_memory, batch_span_context, sent_num
                )
                # prepare label for candidate spans
                batch_span_label = get_batch_span_label(
                    num_spans, cur_span_idx_set, batch_span_context.device
                )
                # calculate loss
                cur_field_cls_loss, batch_span_logp = self.get_field_cls_info(
                    event_idx, field_idx, batch_cand_emb,
                    batch_span_label=batch_span_label, train_flag=True
                )

                field_mle_loss_list.append(cur_field_cls_loss)

                # cur_span_idx_set needs to ensure at least one element, None
                for span_idx in cur_span_idx_set:
                    # Teacher-forcing Style Training
                    if span_idx is None:
                        span_context = self.event_tables[event_idx].field_queries[field_idx]
                    else:
                        # TODO: add either batch_cand_emb or batch_span_context to the memory tensor
                        span_context = batch_cand_emb[span_idx].unsqueeze(0)
                    
                    span_context_bank.append(span_context)
                    prev_global_memory_idx_list.append(prev_path2global_memory_idx[prev_path])
                    
                    cur_path = prev_path + (span_idx, )
                    if self.config.use_path_mem:
                        cur_decode_context = torch.cat([prev_decode_context, span_context], dim=0)
                        prev_path2prev_decode_context[cur_path] = cur_decode_context
                        prev_path2global_memory_idx[cur_path] = len(prev_global_memory_idx_list) - 1
                    else:
                        prev_path2prev_decode_context[cur_path] = prev_decode_context
            
            span_context_bank = torch.cat(span_context_bank, dim=0).cuda()
            prev_global_memory_idx = torch.LongTensor(prev_global_memory_idx_list).cuda()
            global_path_memory = self.event_tables[event_idx].rnn_cell(span_context_bank,
                (
                    torch.index_select(global_path_memory[0], dim=0, index=prev_global_memory_idx),
                    torch.index_select(global_path_memory[1], dim=0, index=prev_global_memory_idx)
            ))

        return field_mle_loss_list, global_path_memory

    def get_loss_on_doc(self, doc_fea, doc_span_info, span_context_list, doc_sent_context):
        if len(span_context_list) == 0:
            raise Exception('Error: doc_fea.ex_idx {} does not have valid span'.format(doc_fea.ex_idx))

        batch_span_context = torch.cat(span_context_list, dim=0)
        num_spans = len(span_context_list)
        event_idx2field_idx2pre_path2cur_span_idx_set = doc_span_info.event_dag_info

        # 1. get event type classification loss
        event_cls_loss = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=True)

        # 2. for each event type, get field classification loss
        # Note that including the memory tensor into the computing graph can boost the performance (>1 F1)
        all_field_loss_list = []
        sent_num = doc_sent_context.size(0)
        doc_sent_context = doc_sent_context + self.sent_embedding
        prev_global_path_memory = None
        for event_idx, event_label in enumerate(doc_fea.event_type_labels):
            if event_label == 0:
                # treat all spans as invalid arguments for that event,
                # because we need to use all parameters to support distributed training
                prev_decode_context = doc_sent_context
                num_fields = self.event_tables[event_idx].num_fields
                global_path_memory = self.event_tables[event_idx].rnn_cell(self.event_tables[event_idx].event_query)
                for field_idx in range(num_fields):
                    # conduct reasoning on this field
                    batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                        event_idx, field_idx, prev_decode_context, prev_global_path_memory, global_path_memory, batch_span_context, sent_num
                    )
                    # prepare label for candidate spans
                    batch_span_label = get_batch_span_label(
                        num_spans, set(), batch_span_context.device
                    )
                    # calculate the field loss
                    cur_field_cls_loss, batch_span_logp = self.get_field_cls_info(
                        event_idx, field_idx, batch_cand_emb,
                        batch_span_label=batch_span_label, train_flag=True
                    )
                    # update the memory tensor
                    span_context = self.event_tables[event_idx].field_queries[field_idx]
                    if self.config.use_path_mem:
                        global_path_memory = self.event_tables[event_idx].rnn_cell(span_context, global_path_memory)
                        prev_decode_context = torch.cat([prev_decode_context, span_context], dim=0)

                    all_field_loss_list.append(cur_field_cls_loss)
            else:
                field_idx2pre_path2cur_span_idx_set = event_idx2field_idx2pre_path2cur_span_idx_set[event_idx]
                field_loss_list, global_path_memory = self.get_field_mle_loss_list(
                    doc_sent_context, batch_span_context, event_idx, field_idx2pre_path2cur_span_idx_set,
                    prev_global_path_memory=prev_global_path_memory
                )
                all_field_loss_list += field_loss_list
                if prev_global_path_memory is None:
                    prev_global_path_memory = global_path_memory[0] + self.inter_path_embedding
                else:
                    prev_global_path_memory = torch.cat((prev_global_path_memory, global_path_memory[0] + self.inter_path_embedding), dim=0)

        total_event_loss = event_cls_loss + sum(all_field_loss_list)
        return total_event_loss

    def get_mix_loss(self, doc_sent_loss_list, doc_event_loss_list, doc_span_info_list):
        batch_size = len(doc_span_info_list)
        loss_batch_avg = 1.0 / batch_size
        lambda_1 = self.config.loss_lambda
        lambda_2 = 1 - lambda_1

        doc_ner_loss_list = []
        for doc_sent_loss, doc_span_info in zip(doc_sent_loss_list, doc_span_info_list):
            # doc_sent_loss: Size([num_valid_sents])
            doc_ner_loss_list.append(doc_sent_loss.sum())

        return loss_batch_avg * (lambda_1 * sum(doc_ner_loss_list) + lambda_2 * sum(doc_event_loss_list))

    def get_eval_on_doc(self, doc_fea, doc_span_info, span_context_list, doc_sent_context):
        if len(span_context_list) == 0:
            event_pred_list = []
            event_idx2obj_idx2field_idx2token_tup = []
            event_idx2event_decode_paths = []
            for event_idx in range(len(self.event_type_fields_pairs)):
                event_pred_list.append(0)
                event_idx2obj_idx2field_idx2token_tup.append(None)
                event_idx2event_decode_paths.append(None)

            return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
                doc_span_info, event_idx2event_decode_paths

        batch_span_context = torch.cat(span_context_list, dim=0)

        # 1. get event type prediction
        event_pred_list = self.get_event_cls_info(doc_sent_context, doc_fea, train_flag=False)

        # 2. for each event type, get field prediction
        # the following mappings are all implemented using list index
        event_idx2event_decode_paths = []
        event_idx2obj_idx2field_idx2token_tup = []
        sent_num = doc_sent_context.size(0)
        doc_sent_context = doc_sent_context + self.sent_embedding
        prev_global_path_memory = None
        for event_idx, event_pred in enumerate(event_pred_list):
            if event_pred == 0:
                event_idx2event_decode_paths.append(None)
                event_idx2obj_idx2field_idx2token_tup.append(None)
                continue

            num_fields = self.event_tables[event_idx].num_fields

            prev_path2prev_decode_context = {(): doc_sent_context}
            prev_path2global_memory_idx = {
                (): 0
            }
            global_path_memory = self.event_tables[event_idx].rnn_cell(self.event_tables[event_idx].event_query)

            last_field_paths = [()]  # only record paths of the last field
            for field_idx in range(num_fields):
                cur_paths = []
                span_context_bank = []
                prev_global_memory_idx_list = []
                for prev_path in last_field_paths:  # traverse all previous decoding paths
                    # get decoding context
                    prev_decode_context = prev_path2prev_decode_context[prev_path]
                    # conduct reasoning on this field
                    batch_cand_emb, prev_decode_context = self.conduct_field_level_reasoning(
                        event_idx, field_idx, prev_decode_context, prev_global_path_memory, global_path_memory, batch_span_context, sent_num
                    )

                    # get field prediction for all spans
                    span_pred_list, _ = self.get_field_cls_info(
                        event_idx, field_idx, batch_cand_emb, train_flag=False
                    )

                    # prepare span_idx to be used for the next field
                    cur_span_idx_list = []
                    for span_idx, span_pred in enumerate(span_pred_list):
                        if span_pred == 1:
                            cur_span_idx_list.append(span_idx)
                    if len(cur_span_idx_list) == 0:
                        # all span is invalid for this field, just choose 'Unknown' token
                        cur_span_idx_list.append(None)

                    for span_idx in cur_span_idx_list:
                        if span_idx is None:
                            span_context = self.event_tables[event_idx].field_queries[field_idx]
                            # span_context = none_span_context
                        else:
                            span_context = batch_cand_emb[span_idx].unsqueeze(0)

                        span_context_bank.append(span_context)
                        prev_global_memory_idx_list.append(prev_path2global_memory_idx[prev_path])
                    
                        cur_path = prev_path + (span_idx, )
                        cur_decode_context = torch.cat([prev_decode_context, span_context], dim=0)
                        prev_path2global_memory_idx[cur_path] = len(prev_global_memory_idx_list) - 1
                        cur_paths.append(cur_path)
                        prev_path2prev_decode_context[cur_path] = cur_decode_context

                # update decoding paths
                last_field_paths = cur_paths
                span_context_bank = torch.cat(span_context_bank, dim=0).cuda()
                prev_global_memory_idx = torch.LongTensor(prev_global_memory_idx_list).cuda()
                global_path_memory = self.event_tables[event_idx].rnn_cell(span_context_bank,
                    (
                        torch.index_select(global_path_memory[0], dim=0, index=prev_global_memory_idx),
                        torch.index_select(global_path_memory[1], dim=0, index=prev_global_memory_idx)
                ))
            
            if prev_global_path_memory is None:
                prev_global_path_memory = global_path_memory[0] + self.inter_path_embedding
            else:
                prev_global_path_memory = torch.cat((prev_global_path_memory, global_path_memory[0] + self.inter_path_embedding), dim=0)

            obj_idx2field_idx2token_tup = []
            for decode_path in last_field_paths:
                assert len(decode_path) == num_fields
                field_idx2token_tup = []
                for span_idx in decode_path:
                    if span_idx is None:
                        token_tup = None
                    else:
                        token_tup = doc_span_info.span_token_tup_list[span_idx]

                    field_idx2token_tup.append(token_tup)
                obj_idx2field_idx2token_tup.append(field_idx2token_tup)

            event_idx2event_decode_paths.append(last_field_paths)
            event_idx2obj_idx2field_idx2token_tup.append(obj_idx2field_idx2token_tup)

        # the first three terms are for metric calculation, the last two are for case studies
        return doc_fea.ex_idx, event_pred_list, event_idx2obj_idx2field_idx2token_tup, \
            doc_span_info, event_idx2event_decode_paths

    def get_local_context_info(self, doc_batch_dict, train_flag=False, use_gold_span=False):
        label_key = 'doc_token_labels'
        if train_flag or use_gold_span:
            assert label_key in doc_batch_dict
            need_label_flag = True
        else:
            need_label_flag = False

        if need_label_flag:
            doc_token_labels_list = doc_batch_dict[label_key]
        else:
            doc_token_labels_list = None

        batch_size = len(doc_batch_dict['ex_idx'])
        doc_token_ids_list = doc_batch_dict['doc_token_ids']
        doc_token_masks_list = doc_batch_dict['doc_token_masks']
        valid_sent_num_list = doc_batch_dict['valid_sent_num']

        # transform doc_batch into sent_batch
        ner_batch_idx_start_list = [0]
        ner_token_ids = []
        ner_token_masks = []
        ner_token_labels = [] if need_label_flag else None
        for batch_idx, valid_sent_num in enumerate(valid_sent_num_list):
            idx_start = ner_batch_idx_start_list[-1]
            idx_end = idx_start + valid_sent_num
            ner_batch_idx_start_list.append(idx_end)

            ner_token_ids.append(doc_token_ids_list[batch_idx])
            ner_token_masks.append(doc_token_masks_list[batch_idx])
            if need_label_flag:
                ner_token_labels.append(doc_token_labels_list[batch_idx])

        # [ner_batch_size, norm_sent_len]  batch每一个是一个句子
        ner_token_ids = torch.cat(ner_token_ids, dim=0)
        ner_token_masks = torch.cat(ner_token_masks, dim=0)
        if need_label_flag:
            ner_token_labels = torch.cat(ner_token_labels, dim=0)

        # get ner output
        ner_token_emb, ner_loss, ner_token_preds = self.ner_model(
            ner_token_ids, ner_token_masks, label_ids=ner_token_labels,
            train_flag=train_flag, decode_flag=not use_gold_span,
        )

        if use_gold_span:  # definitely use gold span info
            ner_token_types = ner_token_labels
        else:
            ner_token_types = ner_token_preds

        # get sentence embedding
        ner_sent_emb = self.get_batch_sent_emb(ner_token_emb, ner_token_masks, valid_sent_num_list)

        assert sum(valid_sent_num_list) == ner_token_emb.size(0) == ner_sent_emb.size(0)

        # followings are all lists of tensors
        doc_token_emb_list = []
        doc_token_masks_list = []
        doc_token_types_list = []
        doc_sent_emb_list = []
        doc_sent_loss_list = []
        for batch_idx in range(batch_size):
            idx_start = ner_batch_idx_start_list[batch_idx]
            idx_end = ner_batch_idx_start_list[batch_idx+1]
            doc_token_emb_list.append(ner_token_emb[idx_start:idx_end, :, :])
            doc_token_masks_list.append(ner_token_masks[idx_start:idx_end, :])
            doc_token_types_list.append(ner_token_types[idx_start:idx_end, :])
            doc_sent_emb_list.append(ner_sent_emb[idx_start:idx_end, :])
            if ner_loss is not None:
                # every doc_sent_loss.size is [valid_sent_num]
                doc_sent_loss_list.append(ner_loss[idx_start:idx_end])

        return doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list

    def forward(self, doc_batch_dict, doc_features,
                train_flag=True, use_gold_span=False, teacher_prob=1,
                event_idx2entity_idx2field_idx=None):
        # Using scheduled sampling to gradually transit to predicted entity spans
        if train_flag and self.config.use_scheduled_sampling:
            # teacher_prob will gradually decrease outside
            if random.random() < teacher_prob:
                use_gold_span = True
            else:
                use_gold_span = False

        # get doc token-level local context
        doc_token_emb_list, doc_token_masks_list, doc_token_types_list, doc_sent_emb_list, doc_sent_loss_list = \
            self.get_local_context_info(
                doc_batch_dict, train_flag=train_flag, use_gold_span=use_gold_span,
            )

        # get doc feature objects
        ex_idx_list = doc_batch_dict['ex_idx']
        doc_fea_list = [doc_features[ex_idx] for ex_idx in ex_idx_list]

        # get doc span-level info for event extraction
        doc_span_info_list = get_doc_span_info_list(doc_token_types_list, doc_fea_list, use_gold_span=use_gold_span)

        # HACK
        graphs = []
        node_features = []
        for idx, doc_span_info in enumerate(doc_span_info_list):
            sent2mention_id = defaultdict(list)
            d = defaultdict(list)

            # 1. sentence-sentence
            node_feature = doc_sent_emb_list[idx] # sent_num * hidden_size
            node_feature += self.sent_embedding
            sent_num = node_feature.size(0)
            for i in range(node_feature.size(0)):
                for j in range(node_feature.size(0)):
                    if i != j:
                        d[('node', 's-s', 'node')].append((i, j))
            
            # 2. sentence-mention
            doc_mention_emb = []
            for mention_id, (sent_idx, char_s, char_e) in enumerate(doc_span_info.mention_drange_list):
                mention_id += sent_num
                mention_token_emb = doc_token_emb_list[idx][sent_idx, char_s: char_e, :]  # [num_mention_tokens, hidden_size]
                if self.config.seq_reduce_type == 'AWA':
                    mention_emb = self.span_token_reducer(mention_token_emb)  # [hidden_size]
                elif self.config.seq_reduce_type == 'MaxPooling':
                    mention_emb = mention_token_emb.max(dim=0)[0]
                elif self.config.seq_reduce_type == 'MeanPooling':
                    mention_emb = mention_token_emb.mean(dim=0)
                else:
                    raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))
                doc_mention_emb.append(mention_emb.unsqueeze(0))

                sent2mention_id[sent_idx].append(mention_id)
                d[('node', 's-m', 'node')].append((mention_id, sent_idx))
                d[('node', 's-m', 'node')].append((sent_idx, mention_id))

            doc_mention_emb = torch.cat(doc_mention_emb, dim=0)
            # add sentence position embedding
            mention_sent_id_list = [drange[0] for drange in doc_span_info.mention_drange_list]
            doc_mention_emb = self.sent_pos_encoder(doc_mention_emb, sent_pos_ids=mention_sent_id_list)
            doc_mention_emb = self.ment_type_encoder(doc_mention_emb, doc_span_info.mention_type_list)
            doc_mention_emb += self.mention_embedding
            node_feature = torch.cat((node_feature, doc_mention_emb), dim=0)

            # 3. intra
            for _, mention_id_list in sent2mention_id.items():
                for i in range(len(mention_id_list)):
                    for j in range(len(mention_id_list)):
                        if i != j:
                            d[('node', 'm-m', 'node')].append((i, j))
            
            # 4. inter
            for mention_id_b, mention_id_e in doc_span_info.span_mention_range_list:
                for i in range(mention_id_b+sent_num, mention_id_e+sent_num):
                    for j in range(mention_id_b+sent_num, mention_id_e+sent_num):
                        if i != j:
                            d[('node', 'm-m', 'node')].append((i, j))

            # 5. default, when lacking of one of the above four kinds edges
            for rel in self.rel_name_lists:
                if ('node', rel, 'node') not in d:
                    d[('node', rel, 'node')].append((0,0))
                    print('add edge: {}'.format(rel))

            # final
            graph = dgl.heterograph(d)
            graphs.append(graph)
            node_features.append(node_feature)

        # batch
        node_features_big = torch.cat(node_features, dim=0)
        graph_big = dgl.batch(graphs).to(node_features_big.device)
        feature_bank = [node_features_big]
        # with residual connection
        for GCN_layer in self.GCN_layers:
            node_features_big = GCN_layer(graph_big, {"node": node_features_big})["node"]
            feature_bank.append(node_features_big)
        feature_bank = torch.cat(feature_bank, dim=-1)
        node_features_big = self.middle_layer(feature_bank)

        # unbatch
        graphs = dgl.unbatch(graph_big)
        cur_idx = 0
        doc_span_context_list = []
        doc_sent_context_list = []
        for idx, graph in enumerate(graphs):
            sent_num = doc_sent_emb_list[idx].size(0)
            node_num = graphs[idx].number_of_nodes('node')
            doc_sent_context_list.append(node_features_big[cur_idx:cur_idx+sent_num])

            span_context_list = []
            mention_context = node_features_big[cur_idx+sent_num:cur_idx+node_num]
            for mid_s, mid_e in doc_span_info_list[idx].span_mention_range_list: 
                multi_ment_context = mention_context[mid_s:mid_e]
                if self.config.seq_reduce_type == 'AWA':
                        span_context = self.span_mention_reducer(multi_ment_context, keepdim=True)
                elif self.config.seq_reduce_type == 'MaxPooling':
                    span_context = multi_ment_context.max(dim=0, keepdim=True)[0]
                elif self.config.seq_reduce_type == 'MeanPooling':
                    span_context = multi_ment_context.mean(dim=0, keepdim=True)
                else:
                    raise Exception('Unknown seq_reduce_type {}'.format(self.config.seq_reduce_type))

                span_context_list.append(span_context)
            doc_span_context_list.append(span_context_list)
            cur_idx += node_num

        if train_flag:
            doc_event_loss_list = []
            for batch_idx, ex_idx in enumerate(ex_idx_list):
                doc_event_loss_list.append(
                    self.get_loss_on_doc(
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                        span_context_list=doc_span_context_list[batch_idx],
                        doc_sent_context=doc_sent_context_list[batch_idx]
                    )
                )

            mix_loss = self.get_mix_loss(doc_sent_loss_list, doc_event_loss_list, doc_span_info_list)

            return mix_loss
        else:
            # return a list object may not be supported by torch.nn.parallel.DataParallel
            # ensure to run it under the single-gpu mode
            eval_results = []

            for batch_idx, ex_idx in enumerate(ex_idx_list):
                eval_results.append(
                    self.get_eval_on_doc(
                        doc_fea_list[batch_idx],
                        doc_span_info_list[batch_idx],
                        span_context_list=doc_span_context_list[batch_idx],
                        doc_sent_context=doc_sent_context_list[batch_idx]
                    )
                )

            return eval_results

def append_all_spans(last_token_path_list, field_idx, field_idx2span_token_tup2dranges):
    new_token_path_list = []
    span_token_tup2dranges = field_idx2span_token_tup2dranges[field_idx]

    for last_token_path in last_token_path_list:
        for token_tup in span_token_tup2dranges.keys():
            new_token_path = list(last_token_path)
            new_token_path.append(token_tup)
            new_token_path_list.append(new_token_path)

        if len(span_token_tup2dranges) == 0:  # ensure every last path will be extended
            new_token_path = list(last_token_path)
            new_token_path.append(None)
            new_token_path_list.append(new_token_path)

    return new_token_path_list

class AttentiveReducer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentiveReducer, self).__init__()

        self.hidden_size = hidden_size
        self.att_norm = math.sqrt(self.hidden_size)

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.att = None

        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_token_emb, masks=None, keepdim=False):
        # batch_token_emb: Size([*, seq_len, hidden_size])
        # masks: Size([*, seq_len]), 1: normal, 0: pad

        query = self.fc.weight
        if masks is None:
            att_mask = None
        else:
            att_mask = masks.unsqueeze(-2)  # [*, 1, seq_len]

        # batch_att_emb: Size([*, 1, hidden_size])
        # self.att: Size([*, 1, seq_len])
        batch_att_emb, self.att = transformer.attention(
            query, batch_token_emb, batch_token_emb, mask=att_mask
        )

        batch_att_emb = self.dropout(self.layer_norm(batch_att_emb))

        if keepdim:
            return batch_att_emb
        else:
            return batch_att_emb.squeeze(-2)

    def extra_repr(self):
        return 'hidden_size={}, att_norm={}'.format(self.hidden_size, self.att_norm)

class SentencePosEncoder(nn.Module):
    def __init__(self, hidden_size, max_sent_num=100, dropout=0.1):
        super(SentencePosEncoder, self).__init__()

        self.embedding = nn.Embedding(max_sent_num, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_elem_emb, sent_pos_ids=None):
        if sent_pos_ids is None:
            num_elem = batch_elem_emb.size(-2)
            sent_pos_ids = torch.arange(
                num_elem, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )
        elif not isinstance(sent_pos_ids, torch.Tensor):
            sent_pos_ids = torch.tensor(
                sent_pos_ids, dtype=torch.long, device=batch_elem_emb.device, requires_grad=False
            )

        batch_pos_emb = self.embedding(sent_pos_ids)
        out = batch_elem_emb + batch_pos_emb
        out = self.dropout(self.layer_norm(out))

        return out

class MentionTypeEncoder(nn.Module):
    def __init__(self, hidden_size, num_ment_types, dropout=0.1):
        super(MentionTypeEncoder, self).__init__()

        self.embedding = nn.Embedding(num_ment_types, hidden_size)
        self.layer_norm = transformer.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_mention_emb, mention_type_ids):
        if not isinstance(mention_type_ids, torch.Tensor):
            mention_type_ids = torch.tensor(
                mention_type_ids, dtype=torch.long, device=batch_mention_emb.device, requires_grad=False
            )

        batch_mention_type_emb = self.embedding(mention_type_ids)
        out = batch_mention_emb + batch_mention_type_emb
        out = self.dropout(self.layer_norm(out))

        return out

class EventTable(nn.Module):
    def __init__(self, event_type, field_types, hidden_size):
        super(EventTable, self).__init__()

        self.event_type = event_type
        self.field_types = field_types
        self.num_fields = len(field_types)
        self.hidden_size = hidden_size

        self.event_cls = nn.Linear(hidden_size, 2)  # 0: NA, 1: trigger this event
        self.field_cls_list = nn.ModuleList(
            # 0: NA, 1: trigger this field
            [nn.Linear(hidden_size, 2) for _ in range(self.num_fields)]
        )

        # used to aggregate sentence and span embedding
        self.event_query = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for fields that do not contain any valid span
        # self.none_span_emb = nn.Parameter(torch.Tensor(1, self.hidden_size))
        # used for aggregating history filled span info
        self.field_queries = nn.ParameterList(
            [nn.Parameter(torch.Tensor(1, self.hidden_size)) for _ in range(self.num_fields)]
        )

        self.reset_parameters()

        self.rnn_cell = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.event_query.data.uniform_(-stdv, stdv)
        # self.none_span_emb.data.uniform_(-stdv, stdv)
        for fq in self.field_queries:
            fq.data.uniform_(-stdv, stdv)

    def forward(self, sent_context_emb=None, batch_span_emb=None, field_idx=None):
        assert (sent_context_emb is None) ^ (batch_span_emb is None)

        if sent_context_emb is not None:  # [num_spans+num_sents, hidden_size]
            # doc_emb.size = [1, hidden_size]
            doc_emb, _ = transformer.attention(self.event_query, sent_context_emb, sent_context_emb)
            doc_pred_logits = self.event_cls(doc_emb)
            doc_pred_logp = F.log_softmax(doc_pred_logits, dim=-1)

            return doc_pred_logp

        if batch_span_emb is not None:
            assert field_idx is not None
            # span_context_emb: [batch_size, hidden_size] or [hidden_size]
            if batch_span_emb.dim() == 1:
                batch_span_emb = batch_span_emb.unsqueeze(0)
            span_pred_logits = self.field_cls_list[field_idx](batch_span_emb)
            span_pred_logp = F.log_softmax(span_pred_logits, dim=-1)

            return span_pred_logp

    def extra_repr(self):
        return 'event_type={}, num_fields={}, hidden_size={}'.format(
            self.event_type, self.num_fields, self.hidden_size
        )

class MLP(nn.Module):
    """Implements Multi-layer Perception."""

    def __init__(self, input_size, output_size, mid_size=None, num_mid_layer=1, dropout=0.1):
        super(MLP, self).__init__()

        assert num_mid_layer >= 1
        if mid_size is None:
            mid_size = input_size

        self.input_fc = nn.Linear(input_size, mid_size)
        self.out_fc = nn.Linear(mid_size, output_size)
        if num_mid_layer > 1:
            self.mid_fcs = nn.ModuleList(
                nn.Linear(mid_size, mid_size) for _ in range(num_mid_layer-1)
            )
        else:
            self.mid_fcs = []
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.input_fc(x)))
        for mid_fc in self.mid_fcs:
            x = self.dropout(F.relu(mid_fc(x)))
        x = self.out_fc(x)
        return x

def get_span_mention_info(span_dranges_list, doc_token_type_list):
    span_mention_range_list = []
    mention_drange_list = []
    mention_type_list = []
    for span_dranges in span_dranges_list:
        ment_idx_s = len(mention_drange_list)
        for drange in span_dranges:
            mention_drange_list.append(drange)
            sent_idx, char_s, char_e = drange
            mention_type_list.append(doc_token_type_list[sent_idx][char_s])
        ment_idx_e = len(mention_drange_list)
        span_mention_range_list.append((ment_idx_s, ment_idx_e))

    return span_mention_range_list, mention_drange_list, mention_type_list

def extract_doc_valid_span_info(doc_token_type_mat, doc_fea):
    doc_token_id_mat = doc_fea.doc_token_ids.tolist()
    doc_token_mask_mat = doc_fea.doc_token_masks.tolist()

    # [(token_id_tuple, (sent_idx, char_s, char_e)), ...]
    span_token_drange_list = []

    valid_sent_num = doc_fea.valid_sent_num
    for sent_idx in range(valid_sent_num):
        seq_token_id_list = doc_token_id_mat[sent_idx]
        seq_token_mask_list = doc_token_mask_mat[sent_idx]
        seq_token_type_list = doc_token_type_mat[sent_idx]
        seq_len = len(seq_token_id_list)

        char_s = 0
        while char_s < seq_len:
            if seq_token_mask_list[char_s] == 0:
                break

            entity_idx = seq_token_type_list[char_s]

            if entity_idx % 2 == 1:
                char_e = char_s + 1
                while char_e < seq_len and seq_token_mask_list[char_e] == 1 and \
                        seq_token_type_list[char_e] == entity_idx + 1:
                    char_e += 1

                token_tup = tuple(seq_token_id_list[char_s:char_e])
                drange = (sent_idx, char_s, char_e)

                span_token_drange_list.append((token_tup, drange))

                char_s = char_e
            else:
                char_s += 1

    span_token_drange_list.sort(key=lambda x: x[-1])  # sorted by drange = (sent_idx, char_s, char_e)
    # drange is exclusive and sorted
    token_tup2dranges = OrderedDict()
    for token_tup, drange in span_token_drange_list:
        if token_tup not in token_tup2dranges:
            token_tup2dranges[token_tup] = []
        token_tup2dranges[token_tup].append(drange)

    span_token_tup_list = list(token_tup2dranges.keys())
    span_dranges_list = list(token_tup2dranges.values())

    return span_token_tup_list, span_dranges_list

def get_batch_span_label(num_spans, cur_span_idx_set, device):
    # prepare span labels for this field and this path
    span_field_labels = [
        1 if span_idx in cur_span_idx_set else 0 for span_idx in range(num_spans)
    ]

    batch_field_label = torch.tensor(
        span_field_labels, dtype=torch.long, device=device, requires_grad=False
    )  # [num_spans], val \in {0, 1}

    return batch_field_label
