"""Build an unwrapped inference model from a full training state dictionary."""
from copy import deepcopy
from transformers import AutoModelForCausalLM
from export_student import student_backbone_state
from lotus import Lotus


def build_inference_model(backbone_config, training_state, *, looped, latent_id, start_id,
                          end_id, eos_id, pad_id, c_thought, latent_injection_mode="add"):
    state = student_backbone_state(training_state, cot=not looped)
    embedding_key = next(k for k in state if k.endswith(("wte.weight", "embed_tokens.weight")))
    config = deepcopy(backbone_config)
    config.vocab_size = state[embedding_key].shape[0]
    base = AutoModelForCausalLM.from_config(config, torch_dtype=state[embedding_key].dtype)
    base.load_state_dict(state, strict=True)
    if looped:
        return Lotus(base, latent_id, start_id, end_id, eos_id, pad_token_id=pad_id,
                     c_thought=c_thought, latent_injection_mode=latent_injection_mode).eval()
    return base.eval()
