import torch

class MultiheadFlashAttentionLayer(torch.nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout, block_size=256, qkv_bias=False):
        super().__init__()

        assert(d_out % num_heads == 0), \
            "num_head muss be divisable with d_out" 

        self.num_heads = num_heads
        self.head_dim = d_out // self.num_heads
        self.block_size = block_size
        self.d_out = d_out

        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_K = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = dropout
        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1
            )
        )
        self.scale = self.head_dim ** 0.5

    def forward(self, inputs: torch.Tensor):
        b, tokens, d_in = inputs.shape
        # calculate Q, K, V
        Q = self.W_Q(inputs) 
        K = self.W_K(inputs)
        V = self.W_V(inputs)

        Q = Q.view(b, tokens, self.num_heads, self.head_dim)
        K = K.view(b, tokens, self.num_heads, self.head_dim)
        V = V.view(b, tokens, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # manual flash implement 
        """
        output_blocks = []

        for q_start in range(0, tokens, self.block_size):
            q_end = q_start + self.block_size
            maximum_score = torch.full(
                (b, self.num_heads, q_end - q_start),
                -float('inf'),
                device=Q.device
            ) 
            sum_of_exponent = torch.zeros( 
                (b, self.num_heads, q_end - q_start),
                device=Q.device
            )
            accumulate_weight = torch.zeros(
                (b, self.num_heads, q_end - q_start, self.head_dim),
                device = Q.device
            )
            q_block = Q[:, :, q_start:q_start + self.block_size,:]
            for kv_start in range(0, tokens, self.block_size):
                kv_end = kv_start + self.block_size
                if kv_start >= q_end: break
                k_block = K[:, :, kv_start:kv_end, :]
                v_block = V[:, :, kv_start:kv_end, :]

                attn_score = q_block @ k_block.transpose(-2, -1) 
                attn_score = attn_score / self.scale 

                if kv_end > q_start: 
                    mask = self.mask[q_start:q_end, kv_start:kv_end]
                    attn_score.masked_fill_(mask, -float("inf"))

                score_block = attn_score.max(dim=-1).values
                new_score_block = torch.maximum(maximum_score, score_block)
                alpha = torch.exp(
                    maximum_score - new_score_block
                )
                exponent_score = torch.exp(
                    attn_score - new_score_block.unsqueeze(-1)
                )
                sum_of_exponent_block = exponent_score.sum(dim=-1)
                o_block = exponent_score @ v_block
                sum_of_exponent = (
                    alpha * sum_of_exponent + sum_of_exponent_block
                )
                accumulate_weight = (
                    alpha.unsqueeze(-1) * accumulate_weight + o_block
                )
                maximum_score = new_score_block
            
            output = accumulate_weight / sum_of_exponent.unsqueeze(-1)
            output_blocks.append(output)
        
        output = torch.cat(output_blocks, dim=2)
        """
        output = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.dropout
        )

        output = output.transpose(1, 2)
        output = output.contiguous().view(b, tokens, self.d_out)
        output = self.out_proj(output)
        return output




            
                





                



            






        
        