"""Cross-size, pre-answer distillation for a frozen explicit reasoner.

The teacher is deliberately held outside the student's registered modules.
Only the width projections are trainable and included in training checkpoints.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


LATENT_TOKENS = {"<|start-latent|>", "<|end-latent|>", "<|latent|>"}


def assert_shared_tokenizer(student, teacher):
    """Require identical natural-language token IDs, allowing latent additions."""
    s = {k: v for k, v in student.get_vocab().items() if k not in LATENT_TOKENS}
    t = {k: v for k, v in teacher.get_vocab().items() if k not in LATENT_TOKENS}
    if s != t or student.bos_token_id != teacher.bos_token_id or student.eos_token_id != teacher.eos_token_id:
        raise ValueError("Teacher and student must share token IDs, BOS and EOS; cross-tokenizer mapping is not implemented.")


def load_teacher(configs, student_tokenizer, device):
    """Load the teacher's own architecture; missing local weights are fatal."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = getattr(configs, "teacher_model_path", None)
    if source in (None, "None", "none", ""):
        raise ValueError("codi_loss_weight > 0 requires teacher_model_path.")
    path = Path(source).expanduser()
    looks_local = source.startswith(("/", ".", "~")) or path.exists()
    if looks_local and not path.exists():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {path}")
    is_hf = not looks_local or (path.is_dir() and (path / "config.json").exists())
    model_id = getattr(configs, "teacher_model_id", None) or configs.model_id
    revision = getattr(configs, "teacher_revision", None)
    dtype = torch.bfloat16 if configs.bf16 else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        str(path) if looks_local and is_hf else source if is_hf else model_id,
        torch_dtype=dtype,
        revision=revision,
    )
    if not is_hf:
        weight_path = path / "model.pt" if path.is_dir() else path
        if not weight_path.is_file():
            raise FileNotFoundError(f"Teacher weights do not exist: {weight_path}")
        weights = torch.load(weight_path, map_location="cpu", weights_only=True)
        if any(k.startswith("base_causallm.") for k in weights):
            weights = {k.removeprefix("base_causallm."): v for k, v in weights.items() if k.startswith("base_causallm.")}
        embedding_key = next((k for k in weights if k.endswith(("wte.weight", "embed_tokens.weight"))), None)
        if embedding_key is not None:
            teacher.resize_token_embeddings(weights[embedding_key].shape[0])
        teacher.load_state_dict(weights, strict=True)
    tokenizer_id = getattr(configs, "teacher_tokenizer_id", None) or model_id
    tokenizer_revision = getattr(configs, "teacher_tokenizer_revision", None)
    teacher_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
    assert_shared_tokenizer(student_tokenizer, teacher_tokenizer)
    teacher.requires_grad_(False).eval()
    return teacher.to(device)


def layer_pairs(student_layers, teacher_layers, mode="relative_depth"):
    """Hidden state 0 is the embedding; pair only transformer block outputs."""
    if min(student_layers, teacher_layers) < 1:
        raise ValueError("Both models must have transformer blocks.")
    if mode == "final":
        return [(student_layers, teacher_layers)]
    if mode != "relative_depth":
        raise ValueError(f"Unknown cross-size alignment: {mode}")
    # Integer half-up rounding makes the mapping stable across Python versions.
    return [(i, max(1, (2 * i * teacher_layers + student_layers) // (2 * student_layers)))
            for i in range(1, student_layers + 1)]


class BoundaryAlignment(nn.Module):
    def __init__(self, student_config, teacher_config, mode="relative_depth", epsilon=1e-6):
        super().__init__()
        self.pairs = layer_pairs(student_config.num_hidden_layers, teacher_config.num_hidden_layers, mode)
        self.epsilon = float(epsilon)
        if self.epsilon <= 0:
            raise ValueError("distillation_epsilon must be positive.")
        self.projections = nn.ModuleList([
            nn.Linear(student_config.hidden_size, teacher_config.hidden_size, bias=False)
            for _ in self.pairs
        ])

    def forward(self, student_hidden, teacher_hidden, student_positions, teacher_positions):
        if student_positions.numel() == 0:
            zero = sum(p.weight.sum() * 0.0 for p in self.projections)
            return zero, student_positions.new_zeros(())
        batch_indices = torch.arange(student_positions.shape[0], device=student_positions.device)
        per_example = 0.0
        for projection, (s_layer, t_layer) in zip(self.projections, self.pairs):
            s = student_hidden[s_layer][batch_indices, student_positions]
            t = teacher_hidden[t_layer][batch_indices, teacher_positions].detach().float()
            projected = projection(s.to(projection.weight.dtype)).float()
            # CODI-style normalization: teacher activation std across features,
            # per example/layer, rather than normalizing the two vectors separately.
            scale = t.std(dim=-1, unbiased=False).clamp_min(self.epsilon)
            per_example = per_example + (projected - t).abs().mean(dim=-1) / scale
        return (per_example / len(self.pairs)).sum(), student_positions.new_tensor(student_positions.numel())


@dataclass
class ExplicitBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    teacher_input_ids: torch.Tensor
    teacher_attention_mask: torch.Tensor
    teacher_position_ids: torch.Tensor
    teacher_positions: torch.Tensor
    student_positions: torch.Tensor


def reconstruct_explicit_batch(input_ids, attention_mask, answer_labels, replaced_cot_steps,
                               start_id, end_id, pad_id, hidden_offset=0):
    """Restore removed CoT and align strictly BEFORE the first answer label.

    In LOTUS the answer begins with `###`; its first token is already an answer
    label. The aligned position therefore precedes that delimiter, not the first
    numeric answer token. The teacher receives no answer tokens at all.
    """
    if answer_labels is None:
        raise ValueError("Cross-size distillation requires answer_labels; rationale labels cannot identify the answer boundary.")
    rows, labels, prefix_rows, student_positions = [], [], [], []
    for b in range(input_ids.shape[0]):
        active = attention_mask[b].bool().nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            raise ValueError("Empty examples are not supported.")
        answer_indices = ((answer_labels[b] != -100) & attention_mask[b].bool()).nonzero(as_tuple=True)[0]
        if answer_indices.numel() == 0:
            raise ValueError("Every example must have at least one answer label.")
        answer_start = int(answer_indices[0])
        position = answer_start - 1 - hidden_offset
        if position < 0:
            raise ValueError("The pre-answer student state is outside the returned hidden states.")
        student_positions.append(position)
        left, right = int(active[0]), int(active[-1]) + 1
        tokens = input_ids[b, left:right].tolist()
        starts = [i for i, token in enumerate(tokens) if token == start_id]
        ends = [i for i, token in enumerate(tokens) if token == end_id]
        answer_offset = answer_start - left
        if starts or ends:
            if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0] or ends[0] >= answer_offset:
                raise ValueError("Expected exactly one latent region before the answer.")
            start, end = starts[0], ends[0]
            question = tokens[:start]
            removed = []
            if replaced_cot_steps is not None:
                for step in replaced_cot_steps[b]:
                    removed.extend(token for token in step.tolist() if token != pad_id)
            rationale = removed + tokens[end + 1:answer_offset]
        else:
            if replaced_cot_steps is not None and replaced_cot_steps[b].numel():
                raise ValueError("Removed CoT requires latent region markers.")
            # This fallback is only used for inputs with no latent markers.
            first_label = int((answer_labels[b] != -100).nonzero(as_tuple=True)[0][0]) - left
            question, rationale = tokens[:first_label], []
        explicit = question + rationale + tokens[answer_offset:]
        if not question or not (question + rationale):
            raise ValueError("A nonempty question is required.")
        rows.append(explicit)
        labels.append([-100] * len(question) + explicit[len(question):])
        prefix_rows.append(question + rationale)

    def pad(sequences, fill):
        result = input_ids.new_full((len(sequences), max(map(len, sequences))), fill)
        mask = torch.zeros_like(result)
        for i, row in enumerate(sequences):
            result[i, :len(row)] = input_ids.new_tensor(row)
            mask[i, :len(row)] = 1
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        return result, mask, positions

    full_ids, full_mask, full_pos = pad(rows, pad_id)
    full_labels, _, _ = pad(labels, -100)
    teacher_ids, teacher_mask, teacher_pos = pad(prefix_rows, pad_id)
    return ExplicitBatch(full_ids, full_mask, full_pos, full_labels,
                         teacher_ids, teacher_mask, teacher_pos,
                         teacher_mask.sum(-1) - 1, input_ids.new_tensor(student_positions))


def cross_size_losses(student, teacher, alignment, explicit, student_hidden,
                      need_distillation=True, need_student_cot=True):
    """Return separate sums/counts for distributed normalization outside FSDP."""
    zero = explicit.input_ids.new_zeros((), dtype=torch.float32)
    count_zero = explicit.input_ids.new_zeros(())
    kd_sum, kd_count, cot_sum, cot_count = zero, count_zero, zero, count_zero
    if need_distillation:
        if teacher is None or alignment is None or student_hidden is None:
            raise ValueError("Distillation requires teacher, alignment and student hidden states.")
        if explicit.teacher_input_ids.max() >= teacher.get_input_embeddings().num_embeddings:
            raise ValueError("Teacher input contains token IDs outside its embedding vocabulary.")
        teacher.eval()
        with torch.no_grad():
            # Skip the LM head: all-layer activations need no 128k-vocabulary logits.
            teacher_output = teacher.base_model(
                input_ids=explicit.teacher_input_ids,
                attention_mask=explicit.teacher_attention_mask,
                position_ids=explicit.teacher_position_ids,
                use_cache=False, output_hidden_states=True, return_dict=True,
            )
        kd_sum, kd_count = alignment(student_hidden, teacher_output.hidden_states,
                                     explicit.student_positions, explicit.teacher_positions)
    if need_student_cot:
        output = student(input_ids=explicit.input_ids, attention_mask=explicit.attention_mask,
                         position_ids=explicit.position_ids, use_cache=False)
        target = explicit.labels[:, 1:]
        cot_sum = F.cross_entropy(output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                                  target.reshape(-1), ignore_index=-100, reduction="sum")
        cot_count = (target != -100).sum()
    return kd_sum, kd_count, cot_sum, cot_count
