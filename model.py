#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — Kiến trúc CVAE dual-decoder (thiết kế B3) + thành phần loss (A3–A7).

Ba biến thể cho TAM GIÁC ABLATION, dùng chung code qua hai cờ:
    seq2seq : use_z=False, use_gram=False   — baseline (a)
    cvae1   : use_z=True,  use_gram=False   — (b)
    cvae2   : use_z=True,  use_gram=True    — (c) mô hình chính

Nguyên tắc bất đối xứng thông tin (chống posterior collapse, A3/A7):
    - Decoder DỊCH   : attention lên encoder states + z nối vào input MỖI bước
    - Decoder NGỮ PHÁP: CHỈ nhận z (không attention) → gradient của nó
      giữ I(x;z) > 0 — đây là "việc làm thật" của z.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, BOS, EOS = 0, 1, 2, 3   # thống nhất với build_vocab trong train.py


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu,σ²) || N(0,I)) THEO TỪNG CHIỀU, trung bình theo batch (A4).
    Trả về vector (d_z,); mỗi phần tử >= 0 (kiểm bằng assert ở train.py)."""
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)   # (B, d_z)
    return kl.mean(dim=0)


class Encoder(nn.Module):
    """BiGRU → encoder states H (cho attention) + tóm tắt câu → (mu, logvar)."""

    def __init__(self, vocab: int, d_emb: int, d_hid: int, d_z: int, use_z: bool):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb, padding_idx=PAD)
        self.rnn = nn.GRU(d_emb, d_hid, batch_first=True, bidirectional=True)
        self.use_z = use_z
        if use_z:
            self.mu = nn.Linear(2 * d_hid, d_z)
            self.logvar = nn.Linear(2 * d_hid, d_z)   # xuất logσ² (ổn định số học, A5)

    def forward(self, x: torch.Tensor, lens: torch.Tensor):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens.cpu(), batch_first=True, enforce_sorted=False)
        H, hn = self.rnn(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(
            H, batch_first=True, total_length=x.size(1))
        summ = torch.cat([hn[0], hn[1]], dim=-1)      # fwd cuối + bwd đầu (A7)
        if self.use_z:
            return H, summ, self.mu(summ), self.logvar(summ)
        return H, summ, None, None


class Attention(nn.Module):
    """Bahdanau additive (A7): e = vᵀ tanh(W s + U h)."""

    def __init__(self, d_dec: int, d_enc2: int, d_attn: int):
        super().__init__()
        self.Wa = nn.Linear(d_dec, d_attn)
        self.Ua = nn.Linear(d_enc2, d_attn)
        self.va = nn.Linear(d_attn, 1)

    def forward(self, s: torch.Tensor, H: torch.Tensor, mask: torch.Tensor):
        e = self.va(torch.tanh(self.Wa(s).unsqueeze(1) + self.Ua(H))).squeeze(-1)
        e = e.masked_fill(~mask, float("-inf"))
        a = torch.softmax(e, dim=-1)                  # (B, T)
        c = torch.bmm(a.unsqueeze(1), H).squeeze(1)   # (B, d_enc2)
        return c, a


class TransDecoder(nn.Module):
    """Decoder dịch: GRUCell, input mỗi bước = [emb(y_prev); c_t; z].
    Output projection TIE với embedding tiếng Việt (tiết kiệm tham số)."""

    def __init__(self, vocab, d_emb, d_hid, d_enc2, d_z, use_z, d_attn):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb, padding_idx=PAD)
        self.use_z = use_z
        d_in = d_emb + d_enc2 + (d_z if use_z else 0)
        self.cell = nn.GRUCell(d_in, d_hid)
        self.attn = Attention(d_hid, d_enc2, d_attn)
        self.init = nn.Linear(d_enc2 + (d_z if use_z else 0), d_hid)
        self.pre = nn.Linear(d_hid + d_enc2, d_emb)

    def s0(self, summ, z):
        x = torch.cat([summ, z], -1) if self.use_z else summ
        return torch.tanh(self.init(x))

    def _logits(self, s, c):
        h = torch.tanh(self.pre(torch.cat([s, c], -1)))
        return F.linear(h, self.emb.weight)           # weight tying

    def _step_input(self, emb_t, c, z):
        return torch.cat([emb_t, c, z], -1) if self.use_z else torch.cat([emb_t, c], -1)

    def forward_teacher(self, y_in, H, mask, summ, z, p_wd: float):
        if self.training and p_wd > 0:                # word dropout (Bowman, A6)
            drop = (torch.rand(y_in.shape, device=y_in.device) < p_wd) \
                   & (y_in != PAD) & (y_in != BOS)
            y_in = y_in.masked_fill(drop, UNK)
        s = self.s0(summ, z)
        outs = []
        for t in range(y_in.size(1)):
            c, _ = self.attn(s, H, mask)              # attention dùng s_{t-1}
            s = self.cell(self._step_input(self.emb(y_in[:, t]), c, z), s)
            outs.append(self._logits(s, c))
        return torch.stack(outs, dim=1)               # (B, T, V)

    @torch.no_grad()
    def greedy(self, H, mask, summ, z, max_len: int = 60):
        B = H.size(0)
        s = self.s0(summ, z)
        y = torch.full((B,), BOS, dtype=torch.long, device=H.device)
        done = torch.zeros(B, dtype=torch.bool, device=H.device)
        out = []
        for _ in range(max_len):
            c, _ = self.attn(s, H, mask)
            s = self.cell(self._step_input(self.emb(y), c, z), s)
            y = self._logits(s, c).argmax(-1)
            y = y.masked_fill(done, PAD)
            out.append(y)
            done |= y.eq(EOS)
            if bool(done.all()):
                break
        return torch.stack(out, dim=1)                # (B, <=max_len)


class GramDecoder(nn.Module):
    """Decoder ngữ pháp: CHỈ nhận z — sinh chuỗi tag pattern (trình bày B2/B3)."""

    def __init__(self, vocab, d_emb, d_hid, d_z):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb, padding_idx=PAD)
        self.cell = nn.GRUCell(d_emb + d_z, d_hid)
        self.init = nn.Linear(d_z, d_hid)
        self.out = nn.Linear(d_hid, vocab)

    def forward_teacher(self, g_in, z):
        s = torch.tanh(self.init(z))
        outs = []
        for t in range(g_in.size(1)):
            s = self.cell(torch.cat([self.emb(g_in[:, t]), z], -1), s)
            outs.append(self.out(s))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def greedy(self, z, max_len: int = 12):
        B = z.size(0)
        s = torch.tanh(self.init(z))
        g = torch.full((B,), BOS, dtype=torch.long, device=z.device)
        done = torch.zeros(B, dtype=torch.bool, device=z.device)
        out = []
        for _ in range(max_len):
            s = self.cell(torch.cat([self.emb(g), z], -1), s)
            g = self.out(s).argmax(-1)
            g = g.masked_fill(done, PAD)
            out.append(g)
            done |= g.eq(EOS)
            if bool(done.all()):
                break
        return torch.stack(out, dim=1)


class CVAE(nn.Module):
    def __init__(self, cfg: dict, n_ja: int, n_vi: int, n_gr: int):
        super().__init__()
        variant = cfg["variant"]
        assert variant in ("seq2seq", "cvae1", "cvae2"), variant
        self.use_z = variant != "seq2seq"
        self.use_gram = variant == "cvae2"
        self.ls_trans = cfg.get("ls_trans", 0.0)  # .get: config.json cũ không có khóa này
        d_z = cfg["d_z"] if self.use_z else 0
        self.d_z = cfg["d_z"]
        self.enc = Encoder(n_ja, cfg["d_emb"], cfg["enc_hid"], cfg["d_z"], self.use_z)
        self.dec_t = TransDecoder(n_vi, cfg["d_emb"], cfg["dec_hid"],
                                  2 * cfg["enc_hid"], cfg["d_z"], self.use_z, cfg["attn"])
        if self.use_gram:
            self.dec_g = GramDecoder(n_gr, cfg["gr_emb"], cfg["gr_hid"], cfg["d_z"])

    # ---------- mã hóa + reparameterization (A5) ----------
    def encode(self, x, lens, sample: bool):
        H, summ, mu, logvar = self.enc(x, lens)
        mask = x.ne(PAD)
        z = None
        if self.use_z:
            if sample and self.training:
                eps = torch.randn_like(mu)
                z = mu + torch.exp(0.5 * logvar) * eps   # z = μ + σ⊙ε
            else:
                z = mu                                    # eval: z = μ (quyết định B5)
        return H, mask, summ, z, mu, logvar

    @staticmethod
    def _ce(logits, target, ls: float = 0.0):
        n = target.ne(PAD).sum().clamp(min=1)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                             target.reshape(-1), ignore_index=PAD, reduction="sum",
                             label_smoothing=ls)
        return ce / n, n                                  # CE THEO TOKEN (trước khi cân λ)

    def forward(self, batch: dict, p_wd: float, sample_z: bool = True):
        H, mask, summ, z, mu, logvar = self.encode(batch["x"], batch["lens"], sample_z)
        logits_t = self.dec_t.forward_teacher(batch["y_in"], H, mask, summ, z, p_wd)
        # smoothing CHỈ lúc train — dev-eval (model.eval()) vẫn báo CE thật,
        # để metrics.csv so được với run cũ và không làm bẩn NLL/ELBO
        ls = self.ls_trans if self.training else 0.0
        ce_t, n_t = self._ce(logits_t, batch["y_out"], ls)
        out = {"ce_trans": ce_t, "n_tok_t": n_t, "mu": mu}
        if self.use_gram:
            logits_g = self.dec_g.forward_teacher(batch["g_in"], z)
            ce_g, n_g = self._ce(logits_g, batch["g_out"])
            out.update(ce_gram=ce_g, n_tok_g=n_g)
        if self.use_z:
            out["klv"] = kl_per_dim(mu, logvar)           # vector (d_z,)
        return out

    @torch.no_grad()
    def predict(self, x, lens, max_vi: int = 60, max_g: int = 12):
        """Greedy decode cho eval/demo. z = μ (deterministic, tái lập được)."""
        self.eval()
        H, mask, summ, z, mu, _ = self.encode(x, lens, sample=False)
        y = self.dec_t.greedy(H, mask, summ, z, max_vi)
        g = self.dec_g.greedy(z, max_g) if self.use_gram else None
        return y, g, mu
