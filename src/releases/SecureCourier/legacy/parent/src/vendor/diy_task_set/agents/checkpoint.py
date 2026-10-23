"""
Checkpoint saving and loading for SecureCourier actor/critic.

Every checkpoint directory contains:
  actor.pt, critic.pt, target_critic.pt,
  actor_optimizer.pt, critic_optimizer.pt,
  meta.json (step, seed, training_phase, training_seed, vocab_hash)
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    ckpt_dir: Path,
    actor,
    critic,
    target_critic,
    actor_opt,
    critic_opt,
    vocab,
    meta: Dict[str, Any],
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(actor.state_dict(),        ckpt_dir / "actor.pt")
    torch.save(critic.state_dict(),       ckpt_dir / "critic.pt")
    torch.save(target_critic.state_dict(),ckpt_dir / "target_critic.pt")
    torch.save(actor_opt.state_dict(),    ckpt_dir / "actor_optimizer.pt")
    torch.save(critic_opt.state_dict(),   ckpt_dir / "critic_optimizer.pt")

    # Save vocab (small, use pickle)
    with open(ckpt_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Compute vocab hash for integrity check
    vocab_bytes = pickle.dumps(vocab)
    meta["vocab_hash"] = hashlib.md5(vocab_bytes).hexdigest()
    meta["actor_param_hash"]  = _param_hash(actor)
    meta["critic_param_hash"] = _param_hash(critic)

    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_checkpoint(ckpt_dir: Path, actor, critic, target_critic,
                    actor_opt=None, critic_opt=None):
    ckpt_dir = Path(ckpt_dir)
    actor.load_state_dict(       torch.load(ckpt_dir / "actor.pt",         weights_only=True))
    critic.load_state_dict(      torch.load(ckpt_dir / "critic.pt",        weights_only=True))
    target_critic.load_state_dict(torch.load(ckpt_dir / "target_critic.pt",weights_only=True))
    if actor_opt and (ckpt_dir / "actor_optimizer.pt").exists():
        actor_opt.load_state_dict( torch.load(ckpt_dir / "actor_optimizer.pt", weights_only=True))
    if critic_opt and (ckpt_dir / "critic_optimizer.pt").exists():
        critic_opt.load_state_dict(torch.load(ckpt_dir / "critic_optimizer.pt", weights_only=True))

    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)
    with open(ckpt_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    return meta, vocab


def load_actor_only(ckpt_dir: Path, actor, vocab_required=True):
    ckpt_dir = Path(ckpt_dir)
    actor.load_state_dict(torch.load(ckpt_dir / "actor.pt", weights_only=True))
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)
    vocab = None
    if vocab_required:
        with open(ckpt_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
    return meta, vocab


def _param_hash(model: torch.nn.Module) -> str:
    h = hashlib.md5()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()
