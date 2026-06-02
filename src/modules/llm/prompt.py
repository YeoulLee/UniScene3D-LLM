"""Prompt construction for SQA3D + Qwen chat template.

A single ``<scene>`` special token is placed in the user turn as a placeholder for the
visual tokens; the model expands that one token into N projected scene tokens at forward
time (see QwenLLM.merge_visual). Labels supervise the assistant (answer) tokens only.

Both the dataset wrapper (to tokenize) and the model (to splice / know the token id) load
the tokenizer through ``load_qwen_tokenizer`` so the ``<scene>`` id is identical on both
sides.
"""

from transformers import AutoTokenizer

SCENE_TOKEN = "<scene>"
IGNORE_INDEX = -100

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a 3D scene. "
    "You are given the scene and a description of your situation in it. "
    "Answer the question with a short word or phrase."
)


def load_qwen_tokenizer(model_id: str):
    """Load the Qwen tokenizer, register ``<scene>``, and ensure a pad token.

    Returns:
        (tokenizer, scene_token_id)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if SCENE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [SCENE_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    scene_token_id = tokenizer.convert_tokens_to_ids(SCENE_TOKEN)
    return tokenizer, scene_token_id


def build_sqa3d_messages(situation: str, question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """Build the chat message list for one SQA3D example (without the answer)."""
    user = f"{SCENE_TOKEN}\nSituation: {situation.strip()}\nQuestion: {question.strip()}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def build_input_and_labels(tokenizer, situation, question, answer=None,
                           system_prompt=DEFAULT_SYSTEM_PROMPT, max_len=None):
    """Tokenize one example into input_ids (+labels when an answer is given).

    Training (answer given): returns the full prompt+answer ids; labels mask everything
    except the answer tokens. Eval (answer is None): returns the prompt ids with the
    generation prompt appended; labels is None.

    Returns:
        dict with ``input_ids`` (LongTensor [L]) and ``labels`` (LongTensor [L] or None).
    """
    messages = build_sqa3d_messages(situation, question, system_prompt)

    # Prompt portion ends right before the assistant turn (generation prompt appended).
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )

    if answer is None:
        input_ids = list(prompt_ids)
        labels = None
    else:
        full_messages = messages + [{"role": "assistant", "content": answer.strip()}]
        full_ids = tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False,
        )
        input_ids = list(full_ids)
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
        # Guard against a degenerate template where prompt is not a strict prefix.
        if len(labels) != len(input_ids):
            labels = [IGNORE_INDEX] * len(input_ids)

    if max_len is not None and len(input_ids) > max_len:
        # Truncate from the left but never drop the <scene> token.
        scene_id = tokenizer.convert_tokens_to_ids(SCENE_TOKEN)
        keep = input_ids[-max_len:]
        if scene_id not in keep:
            keep = [scene_id] + input_ids[-(max_len - 1):]
            if labels is not None:
                labels = [IGNORE_INDEX] + labels[-(max_len - 1):]
        else:
            if labels is not None:
                labels = labels[-max_len:]
        input_ids = keep

    import torch
    out = {"input_ids": torch.tensor(input_ids, dtype=torch.long)}
    out["labels"] = None if labels is None else torch.tensor(labels, dtype=torch.long)
    return out
