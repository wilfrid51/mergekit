# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only


import torch
import transformers


def monkeypatch_lmeval_shuffle():
    """Monkeypatch lm_eval to shuffle the dataset after downloading."""
    import lm_eval.api.task

    if hasattr(lm_eval.api.task.Task, "_monkey_patched"):
        return

    _old_task_dl = lm_eval.api.task.Task.download

    def _dl_shuffled(self: lm_eval.api.task.Task, *args, **kwargs):
        _old_task_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.Task.download = _dl_shuffled

    _old_ct_dl = lm_eval.api.task.ConfigurableTask.download

    def _ct_dl_shuffled(self, *args, **kwargs):
        _old_ct_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.ConfigurableTask.download = _ct_dl_shuffled

    lm_eval.api.task.Task._monkey_patched = True
    print("monkey has been patched")


def monkeypatch_tqdm(lm_eval: bool = True, mergekit: bool = True):
    """Patch lm_eval & mergekit to use Ray's tqdm for progress bars."""

    from ray.experimental.tqdm_ray import tqdm as tqdm_ray

    def _tqdm_wrap(iterable=None, disable: bool = False, **kwargs):
        if disable:
            if iterable is not None:
                return iterable
            return lambda x: x
        res = tqdm_ray(iterable=iterable, **kwargs, flush_interval_s=1.0)
        res.refresh()
        return res

    def _patch_lm_eval():
        import lm_eval

        if hasattr(lm_eval, "_mk_tqdm_patched"):
            return

        import lm_eval.api.metrics
        import lm_eval.api.model
        import lm_eval.api.task
        import lm_eval.models.huggingface
        import lm_eval.models.vllm_causallms

        for module in (
            lm_eval.models.huggingface,
            lm_eval.models.vllm_causallms,
            lm_eval.api.model,
            lm_eval.api.task,
            lm_eval.api.metrics,
        ):
            setattr(module, "tqdm", _tqdm_wrap)

        lm_eval._mk_tqdm_patched = True

    if lm_eval:
        _patch_lm_eval()

    if mergekit:
        del mergekit

        import mergekit
        import mergekit.graph
        import mergekit.merge
        import mergekit.tokenizer

        fake_module = type("fake_module", (), {"tqdm": staticmethod(_tqdm_wrap)})()

        mergekit.graph.tqdm = fake_module
        mergekit.merge.tqdm = fake_module
        mergekit.tokenizer.tqdm = fake_module


def monkeypatch_lmeval_vllm():
    # HACK: fix crash on some tasks due to unset AUTO_MODEL_CLASS for vLLM
    import lm_eval.models.vllm_causallms

    lm_eval.models.vllm_causallms.VLLM.AUTO_MODEL_CLASS = (
        transformers.AutoModelForCausalLM
    )


def monkeypatch_strip_thinking_tags():
    """Monkeypatch to strip <think>...</think> tags from model outputs before evaluation.
  
    This fixes issues where models generate reasoning chains but the evaluator expects
    just the final answer.
    """
    import re
  
    if hasattr(monkeypatch_strip_thinking_tags, "_patched"):
        return
  
    def _strip_thinking(value: str) -> str:
        """Remove <think>...</think> blocks and return the remaining text."""
        if not isinstance(value, str):
            return value
        # Strip think tags and everything inside them
        value = re.sub(r'<think>.*?</think>', '', value, flags=re.DOTALL)
        return value.strip()
  
    # Patch datasets' encoding to strip thinking tags from string values
    # This catches the error point: trying to convert thinking-tagged output to int
    try:
        from datasets.features.features import encode_nested_example as original_encode_nested
        import datasets.features.features as features_module
      
        def patched_encode_nested_example(schema, obj, level=None):
            # Strip thinking tags from string objects before encoding
            if isinstance(obj, str):
                obj = _strip_thinking(obj)
            return original_encode_nested(schema, obj, level)
      
        features_module.encode_nested_example = patched_encode_nested_example
    except (ImportError, AttributeError):
        pass
  
    # Also patch at the string encoding level to be safe
    try:
        from datasets.features.features import UTF8Feature
      
        # Store original encode_example if it's a method
        if hasattr(UTF8Feature, 'encode_example'):
            original_utf8_encode = UTF8Feature.encode_example
          
            def patched_utf8_encode(self, value):
                if isinstance(value, str):
                    value = _strip_thinking(value)
                return original_utf8_encode(self, value)
          
            UTF8Feature.encode_example = patched_utf8_encode
    except (ImportError, AttributeError):
        pass
  
    # Patch value encoding at the lowest level
    try:
        from datasets.features.features import Value
      
        if hasattr(Value, 'encode_example'):
            original_value_encode = Value.encode_example
          
            def patched_value_encode(self, value):
                if isinstance(value, str):
                    value = _strip_thinking(value)
                return original_value_encode(self, value)
          
            Value.encode_example = patched_value_encode
    except (ImportError, AttributeError):
        pass
  
    monkeypatch_strip_thinking_tags._patched = True


def monkeypatch_qwen_tokenizer():
    """Fix missing all_special_tokens_extended attribute in Qwen2Tokenizer."""
    import transformers
    from transformers import Qwen2Tokenizer
    
    if hasattr(Qwen2Tokenizer, "_mk_all_special_tokens_extended_patched"):
        return
    
    def _get_all_special_tokens_extended(self):
        # Fallback implementation
        return self.all_special_tokens
    
    # Add the missing property
    Qwen2Tokenizer.all_special_tokens_extended = property(_get_all_special_tokens_extended)
    Qwen2Tokenizer._mk_all_special_tokens_extended_patched = True

class NoInit:
    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        (k, u, n) = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        transformers.modeling_utils._init_weights = False
        self.funcs = (k, u, n)

    def __exit__(self, *args):
        (k, u, n) = self.funcs
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = (
            k,
            u,
            n,
        )
        transformers.modeling_utils._init_weights = True
