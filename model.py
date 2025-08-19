import torch
from torch import nn
from transformers import BertModel, BertConfig
from embedding_layer import EmbeddingLayer



class MobilityBERT(nn.Module):
    def __init__(self,
                 num_location_ids,
                 transformer_cfg,          # {hidden_size, hidden_layers, attention_heads, dropout, max_seq_length}
                 embedding_sizes,          # {day, time, dow, weekday, location}
                 delta_embedding_dims,     # (f_dim, h_dim, d_dim)
                 feature_configs,          # {"day": {...}, "time": {...}, ...}
                 embedding_combine_mode="cat"):
        super().__init__()

        # 1) Embedding layer
        self.embedding = EmbeddingLayer(
            num_location_ids=num_location_ids,
            day_embedding_size=embedding_sizes["day"],
            time_embedding_size=embedding_sizes["time"],
            dow_embedding_size=embedding_sizes["dow"],
            weekday_embedding_size=embedding_sizes["weekday"],
            location_embedding_size=embedding_sizes["location"],
            delta_embedding_dims=tuple(delta_embedding_dims),
            feature_configs=feature_configs,
            combine_mode=embedding_combine_mode,
            dropout=transformer_cfg["dropout"]
        )
        emb_out_dim = self.embedding.final_dim  # 최종 임베딩 차원

        # 2) BERT 본체
        self.config = BertConfig(
            vocab_size=1,
            hidden_size=transformer_cfg["hidden_size"],
            num_hidden_layers=transformer_cfg["hidden_layers"],
            num_attention_heads=transformer_cfg["attention_heads"],
            intermediate_size=transformer_cfg["hidden_size"] * 4,
            max_position_embeddings=transformer_cfg["max_seq_length"],
            hidden_act='gelu',
            hidden_dropout_prob=transformer_cfg["dropout"],
            attention_probs_dropout_prob=transformer_cfg["dropout"],
            initializer_range=.02,
            layer_norm_eps=1e-12
        )
        self.bert = BertModel(self.config)

        # 3) 투영 & 출력
        # ⛳️ bias=False: 미래 location=0의 "0 기여" 보존
        self.input_projection  = nn.Linear(emb_out_dim, self.config.hidden_size, bias=False)
        self.output_projection = nn.Linear(self.config.hidden_size, num_location_ids)
        self.dropout = nn.Dropout(transformer_cfg["dropout"])

    def forward(self, input_seq_feature, historical_locations, predict_seq_feature):
        """
        input_seq_feature : (B, T_hist, 5)  [day, time, dow, weekday, delta]
        historical_locations: (B, T_hist)
        predict_seq_feature : (B, T_fut , 5)
        """
        # a) 임베딩 (과거/미래) : 미래 location은 0, LN 이후 재마스킹 반영
        hist_embed, fut_embed = self.embedding(
            hist_seq=input_seq_feature,
            hist_loc=historical_locations,
            future_seq=predict_seq_feature
        )  # (B, T_hist, E), (B, T_fut, E)

        # b) 결합 → 투영(첫 선형 bias=False) → BERT
        x = torch.cat([hist_embed, fut_embed], dim=1)           # (B, T_hist+T_fut, E)
        x = self.dropout(self.input_projection(x))              # (B, T, H)
        out = self.bert(inputs_embeds=x).last_hidden_state      # (B, T, H)

        # c) 미래 구간 로짓만 반환
        T_fut = predict_seq_feature.size(1)
        logits = self.output_projection(out[:, -T_fut:])        # (B, T_fut, V)
        return logits