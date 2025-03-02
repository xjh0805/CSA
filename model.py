import torch.nn.init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import heapq
from heapq import nlargest


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X



def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)



class EncoderText(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions"""
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb, cap_len


class AdaptiveSemanticFilter(nn.Module):
    def __init__(self, z1=0.4, z2=0.2, epsilon=1e-9):
        """
        Adaptive Semantic Filter (ASF) Module
        - Filters weak semantic associations based on probability density distributions.

        Args:
            z1 (float): Threshold for Task 1 (scale-balanced alignment).
            z2 (float): Threshold for Task 2 (scale-unbalanced alignment).
            epsilon (float): Small constant to avoid division by zero.
        """
        super(AdaptiveSemanticFilter, self).__init__()
        self.z1 = z1
        self.z2 = z2
        self.epsilon = epsilon

    def compute_similarity_matrix(self, visual_units, textual_units):

        # Compute dot-product similarity using batch matrix multiplication.
        similarity_matrix = torch.bmm(visual_units, textual_units.transpose(1, 2))  # (B, L1, L2)

        # Compute the L2 norm for both visual and textual features along the feature dimension.
        norm_v = torch.norm(visual_units, dim=2, keepdim=True)  # (B, L1, 1)
        norm_t = torch.norm(textual_units, dim=2, keepdim=True)  # (B, L2, 1)
        norm_t = norm_t.transpose(1, 2)  # (B, 1, L2)

        return similarity_matrix / (norm_v * norm_t + self.epsilon)

    def compute_thresholds(self, similarity_matrix):

        device = similarity_matrix.device
        # Use only positive similarity values for computing mu1 and sigma1.
        positive_vals = similarity_matrix[similarity_matrix > 0]
        if positive_vals.numel() > 0:
            mu1 = positive_vals.mean()
            sigma1 = positive_vals.std()
        else:
            mu1 = torch.tensor(0.0, device=device)
            sigma1 = torch.tensor(0.0, device=device)
        mu2 = similarity_matrix.mean()
        sigma2 = similarity_matrix.std()

        # Create tensors for z1 and z2 on the same device.
        z1_tensor = torch.tensor(self.z1, device=device)
        z2_tensor = torch.tensor(self.z2, device=device)

        # Compute thresholds while avoiding log(0) by adding epsilon.
        b1 = mu1 + sigma1 * torch.sqrt(-2 * torch.log(z1_tensor + self.epsilon))
        b2 = mu2 + sigma2 * torch.sqrt(-2 * torch.log(z2_tensor + self.epsilon))
        return b1, b2

    def apply_filter(self, similarity_matrix, visual_units, textual_units):
        """
        Apply the ASF filtering to the similarity matrix.
        """
        b1, b2 = self.compute_thresholds(similarity_matrix)
        B, L1, L2 = similarity_matrix.shape
        device = similarity_matrix.device

        # Initialize masks with epsilon.
        M1 = torch.full((B, L1, L2), self.epsilon, device=device)
        M2 = torch.full((B, L1, L2), self.epsilon, device=device)

        # Iterate over each sample in the batch.
        for i in range(B):
            # If the number of visual and textual units are equal, assume scale-balanced alignment (Task 1).
            if visual_units[i].size(0) == textual_units[i].size(0):
                mask = (similarity_matrix[i] > b1).float()
                M1[i] = mask
            else:
                mask = (similarity_matrix[i] > b2).float()
                M2[i] = mask

        filtered_similarity = similarity_matrix * (M1 + M2).float()
        return filtered_similarity

    def forward(self, visual_units, textual_units):

        similarity_matrix = self.compute_similarity_matrix(visual_units, textual_units)
        filtered_similarity = self.apply_filter(similarity_matrix, visual_units, textual_units)
        return filtered_similarity


def func_attention(query, context, opt, smooth, eps=1e-8, asf_module=None):
    """
    Compute cross-modal attention weights.
    """
    batch_size, queryL = query.size(0), query.size(1)
    _, sourceL = context.size(0), context.size(1)

    # Compute attention scores using batch matrix multiplication.
    queryT = query.transpose(1, 2)  # (B, d, L_query)
    attn = torch.bmm(context, queryT)  # (B, L_context, L_query)

    # Normalize attention scores along the query dimension for each context token.
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, sourceL, queryL)

    # Transpose to get attention scores per query token over context tokens, then apply softmax.
    attn = attn.transpose(1, 2).contiguous()  # (B, L_query, L_context)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * smooth)
    attn = attn.view(batch_size, queryL, sourceL)

    # If an ASF module is provided, filter the attention matrix.
    if asf_module is not None:
        # asf_module should return a filtering matrix of shape (B, L_query, L_context).
        similarity_filter = asf_module(query, context)
        attn = attn * similarity_filter

    # Compute the weighted context based on the attention weights.
    attnT = attn.transpose(1, 2).contiguous()  # (B, L_context, L_query)
    contextT = context.transpose(1, 2)  # (B, d, L_context)
    weightedContext = torch.bmm(contextT, attnT)  # (B, d, L_query)
    weightedContext = weightedContext.transpose(1, 2)  # (B, L_query, d)

    return weightedContext, attnT


def xattn_score_t2i(images, captions, cap_lens, opt,asf_module=None):
    """
    Compute cross-modal similarity between text and image (text-to-image).

    Args:
        images (Tensor): Image features of shape (n_image, L_img, d).
        captions (Tensor): Caption features of shape (n_caption, L_cap, d).
        cap_lens (list or 1D Tensor): Actual lengths of each caption.
        opt (object): Options/config object that must include 'lambda_softmax'.
        asf_module (nn.Module): An instance of the AdaptiveSemanticFilter.

    Returns:
        Tensor: Similarity matrix of shape (n_image, n_caption).
    """
    similarities = []
    n_image, n_caption = images.size(0), captions.size(0)

    for i in range(n_caption):
        n_word = cap_lens[i]
        # Extract the effective tokens for the i-th caption: shape (1, n_word, d)
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # Repeat the caption for each image: shape (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        # Use the caption as the query and images as the context.
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax, asf_module=asf_module)
        # Compute cosine similarity between the original caption and the weighted context,
        # and average over the tokens.
        row_sim = F.cosine_similarity(cap_i_expand, weiContext, dim=2).mean(dim=1, keepdim=True)
        similarities.append(row_sim)

    # Return the final similarity matrix of shape (n_image, n_caption).
    return torch.cat(similarities, dim=1)


def xattn_score_i2t(images, captions, cap_lens, opt,asf_module=None):
    """
    Compute cross-modal similarity between image and text (image-to-text).

    Args:
        images (Tensor): Image features of shape (n_image, L_img, d).
        captions (Tensor): Caption features of shape (n_caption, L_cap, d).
        cap_lens (list or 1D Tensor): Actual lengths of each caption.
        opt (object): Options/config object that must include 'lambda_softmax'.
        asf_module (nn.Module): An instance of the AdaptiveSemanticFilter.

    Returns:
        Tensor: Similarity matrix of shape (n_image, n_caption).
    """
    similarities = []
    n_image, n_caption = images.size(0), captions.size(0)

    for i in range(n_caption):
        n_word = cap_lens[i]
        # Extract the effective tokens for the i-th caption: shape (1, n_word, d)
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # Repeat the caption for each image: shape (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        # Use the image as the query and the caption as the context.
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax, asf_module=asf_module)
        # Compute cosine similarity between the original image and the weighted context,
        # and average over the tokens.
        row_sim = F.cosine_similarity(images, weiContext, dim=2).mean(dim=1, keepdim=True)
        similarities.append(row_sim)

    return torch.cat(similarities, dim=1)


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()



class AdaptiveSemanticAggregation(nn.Module):
    def __init__(self, layers=5, alpha=0.4, top_p=10, window_sizes=[1, 2, 3, 4, 5], steps=[1,1,2,2,3]):
        super().__init__()
        self.layers = layers
        self.alpha = alpha
        self.top_p = top_p
        self.window_sizes = window_sizes
        self.steps = steps

    def position_aware_subsequence(self, token_indices):
        """Generate position-aware subsequences using token indices"""
        subsequences = []
        for w_size, w_step in zip(self.window_sizes, self.steps):
            seq_len = token_indices.size(1)
            for start in range(0, seq_len - w_size + 1, w_step):
                end = start + w_size
                subsequences.append(token_indices[:, start:end])
        return subsequences

    def co_occurrence_aware_subsequence(self, token_indices, co_matrix):
        """Generate co-occurrence-aware subsequences using token indices"""
        batch_size, seq_len = token_indices.size()
        all_subsequences = []
        for b in range(batch_size):
            batch_co_matrix = co_matrix[b]
            subsequences = []
            for i in range(seq_len):
                current_seq = [token_indices[b, i]]
                visited = set([i])
                for _ in range(self.layers):
                    max_score = -1
                    max_idx = -1
                    for j in range(seq_len):
                        if j not in visited and batch_co_matrix[i, j] > max_score:
                            max_score = batch_co_matrix[i, j]
                            max_idx = j
                    if max_score > self.alpha and max_idx != -1:
                        current_seq.append(token_indices[b, max_idx])
                        visited.add(max_idx)
                    else:
                        break
                subsequences.append(torch.tensor(current_seq, device=token_indices.device))
            all_subsequences.extend(subsequences)
        return all_subsequences

    def compute_iou(self, idx_seq1, idx_seq2):
        """Compute IoU based on token index sets"""
        set1 = set(idx_seq1.tolist())
        set2 = set(idx_seq2.tolist())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0

    def batch_compute_iou(self, pos_index_sequences, co_index_sequences):
        """Batch IoU computation for index sequences"""
        pos_sets = [set(seq.tolist()) for seq in pos_index_sequences]
        co_sets = [set(seq.tolist()) for seq in co_index_sequences]

        iou_matrix = torch.zeros(len(pos_sets), len(co_sets), device=pos_index_sequences[0].device)
        for i, pos_set in enumerate(pos_sets):
            for j, co_set in enumerate(co_sets):
                intersection = len(pos_set & co_set)
                union = len(pos_set | co_set)
                iou_matrix[i, j] = intersection / union if union > 0 else 0

        return iou_matrix

    def aggregate_units(self, pos_index_sequences, co_index_sequences, pos_feature_sequences, co_feature_sequences):
        """IoU-based weighted aggregation"""
        iou_matrix = self.batch_compute_iou(pos_index_sequences, co_index_sequences)
        top_p_indices = torch.topk(iou_matrix.flatten(), self.top_p)[1]
        p_indices, c_indices = torch.div(top_p_indices, co_index_sequences[0].size(0),
                                         rounding_mode='floor'), top_p_indices % co_index_sequences[0].size(0)


        iou_weights = iou_matrix[p_indices, c_indices]
        if iou_weights.sum() > 0:
            iou_weights /= iou_weights.sum()
        else:
            iou_weights = torch.ones_like(iou_weights) / len(iou_weights)

        aggregated_units = [torch.cat([pos_feature_sequences[p], co_feature_sequences[c]], dim=0) for p, c in
                            zip(p_indices, c_indices)]
        aggregated_features = torch.stack([
            torch.sum(unit * weight.unsqueeze(-1), dim=0) for unit, weight in zip(aggregated_units, iou_weights)
        ])

        return aggregated_features

    def forward(self, token_indices, co_matrix, token_features):
        """Forward pass with token indices and token features"""
        pos_index_sequences = self.position_aware_subsequence(token_indices)
        co_index_sequences = self.co_occurrence_aware_subsequence(token_indices, co_matrix)

        pos_feature_sequences = self.position_aware_subsequence(token_features)
        co_feature_sequences = self.co_occurrence_aware_subsequence(token_features, co_matrix)

        return self.aggregate_units(pos_index_sequences, co_index_sequences, pos_feature_sequences,
                                    co_feature_sequences)


class CSA(object):
    def __init__(self, opt):
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecomp(opt.img_dim, opt.embed_size, opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers, opt.use_bi_gru, opt.no_txtnorm)
        self.asa_img = AdaptiveSemanticAggregation(opt.layers, opt.alpha, opt.top_p, opt.window_sizes, opt.steps)
        self.asa_txt = AdaptiveSemanticAggregation(opt.layers, opt.alpha, opt.top_p, opt.window_sizes, opt.steps)
        self.asf = AdaptiveSemanticFilter(z1=opt.z1, z2=opt.z2)  # Add ASF module
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.asa_img.cuda()
            self.asa_txt.cuda()
            self.asf.cuda()
            cudnn.benchmark = True
        self.criterion = ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
        self.params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters()) \
                      + list(self.asa_img.parameters()) + list(self.asa_txt.parameters()) \
                      + list(self.asf.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        self.Eiters = 0


    def state_dict(self):
        return [self.img_enc.state_dict(), self.txt_enc.state_dict()]

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, co_matrix_img, co_matrix_txt):
        img_emb = self.img_enc(images)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        img_enhanced = self.asa_img(img_emb, co_matrix_img)
        cap_enhanced = self.asa_txt(cap_emb, co_matrix_txt)
        img_enhanced = l2norm(img_enhanced, dim=-1)
        cap_enhanced = l2norm(cap_enhanced, dim=-1)
        return img_enhanced, cap_enhanced, cap_lens

    def forward_loss(self, img_emb, cap_emb, cap_len):
        loss = self.criterion(img_emb, cap_emb, cap_len)
        return loss

    def train_emb(self, images, captions, lengths, co_matrix_img, co_matrix_txt):
        self.Eiters += 1
        img_emb, cap_emb, cap_lens = self.forward_emb(images, captions, lengths, co_matrix_img, co_matrix_txt)
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens)
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

