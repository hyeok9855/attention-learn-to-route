import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from utils.functions import sample_many

from typing import Optional, Tuple


class LayerNormLSTMCell(nn.Module):
    """
    ## Long Short-Term Memory Cell
    LSTM Cell computes $c$, and $h$. $c$ is like the long-term memory,
    and $h$ is like the short term memory.
    We use the input $x$ and $h$ to update the long term memory.
    In the update, some features of $c$ are cleared with a forget gate $f$,
    and some features $i$ are added through a gate $g$.
    The new short term memory is the $\tanh$ of the long-term memory
    multiplied by the output gate $o$.
    Note that the cell doesn't look at long term memory $c$ when doing the update. It only modifies it.
    Also $c$ never goes through a linear transformation.
    This is what solves vanishing and exploding gradients.
    Here's the update rule.
    \begin{align}
    c_t &= \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t) \\
    h_t &= \sigma(o_t) \odot \tanh(c_t)
    \end{align}
    $\odot$ stands for element-wise multiplication.
    Intermediate values and gates are computed as linear transformations of the hidden
    state and input.
    \begin{align}
    i_t &= lin_x^i(x_t) + lin_h^i(h_{t-1}) \\
    f_t &= lin_x^f(x_t) + lin_h^f(h_{t-1}) \\
    g_t &= lin_x^g(x_t) + lin_h^g(h_{t-1}) \\
    o_t &= lin_x^o(x_t) + lin_h^o(h_{t-1})
    \end{align}
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        # These are the linear layer to transform the `input` and `hidden` vectors.
        # One of them doesn't need a bias since we add the transformations.

        # This combines $lin_x^i$, $lin_x^f$, $lin_x^g$, and $lin_x^o$ transformations.
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        # This combines $lin_h^i$, $lin_h^f$, $lin_h^g$, and $lin_h^o$ transformations.
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

        # Whether to apply layer normalizations.
        #
        # Applying layer normalization gives better results.
        # $i$, $f$, $g$ and $o$ embeddings are normalized and $c_t$ is normalized in
        # $h_t = o_t \odot \tanh(\mathop{LN}(c_t))$
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
        self.layer_norm_c = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # We compute the linear transformations for $i_t$, $f_t$, $g_t$ and $o_t$
        # using the same linear layers.
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        # Each layer produces an output of 4 times the `hidden_size` and we split them
        ifgo = ifgo.chunk(4, dim=-1)

        # Apply layer normalization (not in original paper, but gives better results)
        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]

        # $$i_t, f_t, g_t, o_t$$
        i, f, g, o = ifgo

        # $$c_t = \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t) $$
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        # $$h_t = \sigma(o_t) \odot \tanh(c_t)$$
        # Optionally, apply layer norm to $c_t$
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next


class LayerNormLSTM(nn.Module):
    """
    ## Multilayer LSTM
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 1):
        """
        Create a network of `n_layers` of LSTM.
        """

        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # Create cells for each layer. Note that only the first layer gets the input directly.
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([LayerNormLSTMCell(input_size, hidden_size)] +
                                   [LayerNormLSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        `x` has shape `[n_steps, batch_size, input_size]` and
        `state` is a tuple of $h$ and $c$, each with a shape of `[batch_size, hidden_size]`.
        """
        n_steps, batch_size = x.shape[:2]

        # Initialize the state if `None`
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            # Reverse stack the tensors to get the states of each layer
            #
            # 📝 You can just work with the tensor itself but this is easier to debug
            h, c = list(torch.unbind(h)), list(torch.unbind(c))

        # Array to collect the outputs of the final layer at each time step.
        out = []
        for t in range(n_steps):
            # Input to the first layer is the input itself
            inp = x[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the layer
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                # Input to the next layer is the state of this layer
                inp = h[layer]
            # Collect the output $h$ of the final layer
            out.append(h[-1])

        # Stack the outputs and states
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        return out, (h, c)


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""
    def __init__(self, input_dim, hidden_dim, layer_norm=False):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = (
            LayerNormLSTM(input_dim, hidden_dim) if layer_norm else nn.LSTM(input_dim, hidden_dim)
        )
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden
    
    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        std = 1. / math.sqrt(hidden_dim)
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_hx.data.uniform_(-std, std)

        enc_init_cx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_cx.data.uniform_(-std, std)
        return enc_init_hx, enc_init_cx


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""
    def __init__(self, dim, use_tanh=False, C=10, layer_norm=False):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln_q = nn.LayerNorm(dim)
            self.ln_r = nn.LayerNorm(dim)
        
    def forward(self, query, ref):
        """
        Args: 
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder. 
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query)
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL

        if self.layer_norm:
            q = self.ln_q(q)
            e = self.ln_r(e.permute(2, 0, 1)).permute(1, 2, 0)

        q = q.unsqueeze(2)  # batch x dim x 1

        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2)) 
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
                expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u  
        return e, logits


class Decoder(nn.Module):
    def __init__(self, 
            embedding_dim,
            hidden_dim,
            tanh_exploration,
            use_tanh,
            n_glimpses=1,
            mask_glimpses=True,
            mask_logits=True,
            layer_norm=False,
        ):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = None  # Needs to be set explicitly before use

        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, layer_norm=layer_norm)
        self.glimpse = Attention(hidden_dim, use_tanh=False, layer_norm=layer_norm)
        self.sm = nn.Softmax(dim=1)

    def update_mask(self, mask, selected):
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):

        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        if not self.mask_logits:
            # If self.mask_logits, this would be redundant, otherwise we must mask to make sure we don't resample
            # Note that as a result the vector of probs may not sum to one (this is OK for .multinomial sampling)
            # But practically by not masking the logits, a model is learned over all sequences (also infeasible)
            # while only during sampling feasibility is enforced (a.k.a. by setting to 0. here)
            probs[logit_mask] = 0.
            # For consistency we should also mask out in log_p, but the values set to 0 will not be sampled and
            # Therefore not be used by the reinforce estimator

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[logit_mask] = -np.inf
            # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
            # [batch_size x h_dim x 1]
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf

        return logits, h_out

    def forward(self, decoder_input, embedded_inputs, hidden, context, pi_expert=None):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim]. 
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim] 
        """

        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = Variable(
            embedded_inputs.data.new(embedded_inputs.size(1), embedded_inputs.size(0)).byte().zero_(),
            requires_grad=False
        )

        for i in steps:
            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context)
            # select the next inputs for the decoder [batch_size x hidden_dim]
            idxs = self.decode(probs, mask) if pi_expert is None else pi_expert[:, i]
            idxs = idxs.detach()  # Otherwise pytorch complains it want's a reward, todo implement this more properly?

            # Gather input embedding of selected
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
            ).squeeze(0)

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)

        return (torch.stack(outputs, 1), torch.stack(selections, 1)), hidden

    def decode(self, probs, mask):
        if self.decode_type == "greedy":
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs


class CriticNetworkLSTM(nn.Module):
    """Useful as a baseline in REINFORCE updates"""
    def __init__(self,
            embedding_dim,
            hidden_dim,
            n_process_block_iters,
            tanh_exploration,
            use_tanh):
        super(CriticNetworkLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(embedding_dim, hidden_dim)
        
        self.process_block = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """
        inputs = inputs.transpose(0, 1).contiguous()

        encoder_hx = self.encoder.init_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = self.encoder.init_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        
        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))
        
        # grab the hidden state and process it via the process block 
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out


class PointerNetwork(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=None,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization=None,
                 imitation=False,
                 layer_norm=False,
                 **kwargs):
        super(PointerNetwork, self).__init__()

        self.problem = problem
        assert problem.NAME == "tsp", "Pointer Network only supported for TSP"
        self.input_dim = 2

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            layer_norm=layer_norm,
        )

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            tanh_exploration=tanh_clipping,
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits,
            layer_norm=layer_norm,
        )

        # Trainable initial hidden states
        std = 1. / math.sqrt(embedding_dim)
        self.decoder_in_0 = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.decoder_in_0.data.uniform_(-std, std)

        self.embedding = nn.Parameter(torch.FloatTensor(self.input_dim, embedding_dim))
        self.embedding.data.uniform_(-std, std)

        self.imitation = imitation

    def set_decode_type(self, decode_type, temp=None):
        self.decoder.decode_type = decode_type

    def forward(self, inputs, pi_expert=None, return_pi=False):
        if self.imitation:
            if self.training:
                assert pi_expert is not None, "To train model in imitation manner, pi_expert is required"
            else:
                assert pi_expert is None, "For evaluation, pi_expert will not be used!"

        batch_size, graph_size, input_dim = inputs.size()

        embedded_inputs = torch.mm(
            inputs.transpose(0, 1).contiguous().view(-1, input_dim),
            self.embedding
        ).view(graph_size, batch_size, -1)

        # query the actor net for the input indices 
        # making up the output, and the pointer attn 
        _log_p, pi = self._inner(embedded_inputs, pi_expert)  # pi == pi_expert for imitation learning training

        cost, mask = self.problem.get_costs(inputs, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if return_pi:
            return cost, ll, pi

        return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _inner(self, inputs, pi_expert=None):

        encoder_hx = encoder_cx = Variable(
            torch.zeros(1, inputs.size(1), self.encoder.hidden_dim, out=inputs.data.new()),
            requires_grad=False
        )

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)

        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                                                                 inputs,
                                                                 dec_init_state,
                                                                 enc_h,
                                                                 pi_expert)

        return pointer_probs, input_idxs
    
    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function

        def input_embedding(inputs, embedding_matrix):
            batch_size, graph_size, input_dim = inputs.size()

            embedded_inputs = torch.mm(
                inputs.transpose(0, 1).contiguous().view(-1, input_dim),
                self.embedding
            ).view(graph_size, batch_size, -1)
            return embedded_inputs

        return sample_many(
            lambda input: self._inner(input_embedding(input, self.embedding)),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input, pi),  # Don't need embeddings as input to get_costs
            input,
            batch_rep, iter_rep
        )