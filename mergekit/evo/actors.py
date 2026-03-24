from __future__ import annotations
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import gc
import logging
import tempfile
from typing import Optional, Union

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import lm_eval.tasks
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch
import transformers
from transformers.utils import is_flash_attn_2_available

from mergekit.architecture.base import ConfiguredModelArchitecture

try:
    import vllm
except ImportError:
    vllm = None


from mergekit.architecture import arch_info_for_config
from mergekit.common import get_torch_accelerator_module, get_torch_accelerator_type
from mergekit.config import MergeConfiguration
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.genome import InvalidGenotypeError, ModelGenome
from mergekit.evo.helpers import _eval_model, evaluate_model, merge_model
from mergekit.evo.monkeypatch import (
    NoInit,
    monkeypatch_lmeval_shuffle,
    monkeypatch_lmeval_vllm,
)
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.merge import _model_out_config
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner

LOG = logging.getLogger(__name__)


class MergeActorBase:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        batch_size: Optional[int] = None,
        task_manager: Optional[lm_eval.tasks.TaskManager] = None,
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.cache = LoaderCache()
        self.cache.setup(merge_options)
        self.model_storage_path = model_storage_path
        self.vllm = vllm
        self.batch_size = batch_size
        self.task_manager = task_manager
        self.quantization_config = quantization_config

        if config.shuffle:
            monkeypatch_lmeval_shuffle()

        # monkeypatch_tqdm()
        if self.vllm:
            monkeypatch_lmeval_vllm()


@ray.remote(num_cpus=1, num_gpus=1.0)
class OnDiskMergeEvaluator(MergeActorBase):
    """
    Merges models to disk then evaluates them in a separate process.

    Maximum compatibility and potential for parallelism, but higher overhead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> dict:
        gc.collect()
        torch_accelerator_module = get_torch_accelerator_module(
            self.merge_options.device
        )
        torch_accelerator_module.empty_cache()
        LOG.info("Merging model")
        print("Merging model")
        merged_path = merge_model(
            genotype, self.genome, self.model_storage_path, self.merge_options
        )
        if not merged_path:
            LOG.error("Model merge failed")
            print("Model merge failed")
            return {"score": None, "results": None}

        # Add smoke test here
        # LOG.info("Running smoke test")
        # print("Running smoke test")
        try:
            model_kwargs = {}
            if self.quantization_config is not None:
                model_kwargs["quantization_config"] = self.quantization_config

            # Load model for smoke test
            from lm_eval.models.huggingface import HFLM
            test_model = HFLM(pretrained=merged_path, **model_kwargs)

            # if not self.smoke_test_model(test_model, self.task_manager):
            #     LOG.warning("Smoke test failed - returning zero score")
            #     print("Smoke test failed - returning zero score")
            #     return {"score": 0.0, "results": {task.name: {"acc": 0.0} for task in self.config.tasks}}
            # else:
            #     print("Smoke test passed - Congratulation")
        except Exception as e:
            LOG.error(f"Smoke test error: {e}")
            print(f"Smoke test error: {e}")
            return {"score": 0.0, "results": {task.name: {"acc": 0.0} for task in self.config.tasks}}

        LOG.info(f"Model merged to {merged_path}")
        print(f"Model merged to {merged_path}")
        return evaluate_model(
            merged_path,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            vllm=self.vllm,
            batch_size=self.batch_size,
            task_manager=self.task_manager,
            apply_chat_template=self.config.apply_chat_template,
            fewshot_as_multiturn=self.config.fewshot_as_multiturn,
            model_kwargs=model_kwargs,
        )
  
    def smoke_test_model(self, model, task_manager=None):
        """Quick test to check if model produces coherent responses."""
        try:
            # Simple smoke test prompt
            test_prompt = "Tell me about yourself."

            # Generate response
            # if hasattr(model, 'generate'):
            print("Test prompt: {test_prompt}")
            print(type(model))
            if hasattr(model, "model") and hasattr(model, "tokenizer"):
                hf_model = model.model
                tokenizer = model.tokenizer
                inputs = tokenizer(test_prompt, return_tensors="pt")
                inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = hf_model.generate(
                        **inputs,
                        max_new_tokens=4096,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                logging.warning(f"Unsupported model type for smoke test: {type(model)}")
                return False

            print(f"Current Merged model's response is : {response[:1000]}")

            # Check for rambling / repetition
            if self.detect_rambling_simple(response):
                return False

            if response == "Model response test failed":
                return False

            return True
        except Exception as e:
            logging.warning(f"Smoke test failed: {e}")
            return False

    def detect_rambling_simple(self, text):
        """Simple detection of rambling/repetition."""
        # Check for excessive repetition
        words = text.split()
        rambling_cnt = 0
        if len(words) > 20:
            # Count repeated phrases
            for i in range(len(words) - 10):
                phrase = ' '.join(words[i:i+10])
                if text.count(phrase) > 3:
                    rambling_cnt += 1
                    if rambling_cnt > 3:
                        return True
        return False


@ray.remote(num_cpus=1, num_gpus=1)
class InMemoryMergeEvaluator(MergeActorBase):
    """
    Performs merges in memory, using a single model instance.

    This reduces overhead from disk I/O and model loading, but prevents
    parallelism and may be slower for large models.

    Implementation is dark sorcery tampering with the internals of lm-eval,
    transformers, and vLLM and may break at any time.
    """

    model: object | None = None
    arch_info: Optional[ConfiguredModelArchitecture] = None

    def __init__(
        self,
        *args,
        vllm: bool = False,
        **kwargs,
    ):
        # assert not vllm, "VLLM is not supported for in-memory merging"
        super().__init__(*args, vllm=vllm, **kwargs)

    def _maybe_init_model(self, config: MergeConfiguration):
        ai = arch_info_for_config(self.genome._input_config_example)
        cfg_out = _model_out_config(
            config,
            ai,
            trust_remote_code=self.merge_options.trust_remote_code,
        )
        cfg_out.use_cache = True
        cfg_out.torch_dtype = torch.bfloat16

        if self.arch_info is not None:
            different = False
            for key in cfg_out.to_diff_dict():
                if key in ["architectures", "model_type"]:
                    # to get to here we must have --allow-crimes set, so let it ride
                    continue
                elif key in ["use_cache", "torch_dtype"]:
                    continue
                elif key.endswith("_token_id"):
                    # update our config but don't fail if it's different
                    setattr(self.arch_info.config, key, getattr(cfg_out, key, None))
                    continue

                if getattr(cfg_out, key) != getattr(self.arch_info.config, key, None):
                    LOG.warning(f"Config key {key} changed, reinitializing model")
                    different = True
                    break

            if not different:
                return

        self.inner_model = None

        model_kwargs = {
            "trust_remote_code": self.merge_options.trust_remote_code,
            "torch_dtype": torch.bfloat16,
        }
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        with NoInit():
            inner_model = (
                transformers.AutoModelForCausalLM.from_config(
                    cfg_out,
                    **model_kwargs,
                )
                .bfloat16()
                .to(self.merge_options.device)
                .eval()
                .requires_grad_(False)
            )

        if self.vllm:
            # oh i hate this
            with tempfile.TemporaryDirectory(
                dir=self.model_storage_path, prefix="vllm"
            ) as tempdir:
                inner_model.save_pretrained(
                    tempdir, safe_serialization=True, out_shard_size=1_000_000_000_000
                )
                del inner_model
                tokenizer_donor = self.genome.definition.base_model
                if tokenizer_donor is None:
                    LOG.warning(
                        "Base model not set, using tokenizer from first model in genome"
                    )
                    tokenizer_donor = self.genome.definition.models[0]
                tok = transformers.AutoTokenizer.from_pretrained(
                    tokenizer_donor.model.path, use_fast=True
                )
                tok.save_pretrained(tempdir)

                max_model_len = None
                if (
                    seq_len := getattr(cfg_out, "max_position_embeddings", None)
                ) is not None:
                    max_model_len = seq_len
                if (window_sz := getattr(cfg_out, "sliding_window", None)) is not None:
                    max_model_len = min(max_model_len or 1024, window_sz)
                if max_model_len and max_model_len > 8192:
                    max_model_len = 8192
                    LOG.warning(f"Clipping sequence length to {max_model_len}")

                accelerator_type = get_torch_accelerator_type(self.merge_options.device)
                mem_util = (
                    0.7 if accelerator_type in ["cuda", "xpu"] else 0.9
                )  # reduce memory usage if we're also using accelerator for the merge
                self.model = lm_eval.models.vllm_causallms.VLLM(
                    pretrained=tempdir,
                    batch_size=self.batch_size or "auto",
                    max_model_len=max_model_len,
                    gpu_memory_utilization=mem_util,
                    dtype="bfloat16",
                    device=self.merge_options.device,
                    trust_remote_code=self.merge_options.trust_remote_code,
                )
        else:
            self.model = lm_eval.models.huggingface.HFLM(pretrained=inner_model)
        self.arch_info = (
            ConfiguredModelArchitecture(
                info=ai,
                config=cfg_out,
            )
            if ai
            else None
        )
        LOG.info("Model initialized")

    def evaluate(self, genotype: torch.Tensor) -> dict:
        try:
            config = self.genome.genotype_merge_config(genotype)
        except InvalidGenotypeError as e:
            LOG.error("Invalid genotype", exc_info=e)
            return {"score": None, "results": None}

        self._maybe_init_model(config)

        # Add smoke test before full evaluation
        LOG.info("Running smoke test")
        if not self.smoke_test_model(self.model, self.task_manager):
            LOG.warning("Smoke test failed - returning zero score")
            return {"score": 0.0, "results": {task.name: {"acc": 0.0} for task in self.config.tasks}}
        else:
            LOG.info("Smoke test passed - Congratulation")

        planner = MergePlanner(
            config,
            self.arch_info.info,
            self.merge_options,
            self.arch_info.config,
        )

        tasks = planner.plan_in_memory()

        model = self.model.model
        if vllm is not None and isinstance(model, vllm.LLM):
            assert (
                model.llm_engine.parallel_config.world_size == 1
            ), "Must be single GPU"
            engine = model.llm_engine
            if hasattr(engine, "model_executor"):
                worker = engine.model_executor.worker
            elif hasattr(engine, "driver_worker"):
                worker = engine.driver_worker
            else:
                raise ValueError("Unknown LLM engine type")
            model = worker.model_runner.model
        param_dict = dict(model.named_parameters())

        stacked_mapping = {
            # mappings for Llama/Mistral attention weights to vLLM packed tensors
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        }

        accelerator_type = get_torch_accelerator_type(self.merge_options.device)
        executor = Executor(
            tasks,
            math_device=(
                self.merge_options.device
                if accelerator_type in ["cuda", "xpu"]
                else "cpu"
            ),
            storage_device=(
                self.merge_options.device
                if accelerator_type in ["cuda", "xpu"]
                else "cpu"
            ),
        )
        for tensor_task, value in executor.run(quiet=True):
            assert isinstance(tensor_task, ReturnTensor)
            name = tensor_task.weight_info.name

            if name in param_dict:
                param_dict[name].data.copy_(value, non_blocking=True)
            elif self.vllm:
                stacked = False
                for needle, (replacement, shard_id) in stacked_mapping.items():
                    if needle in name:
                        target = name.replace(needle, replacement)
                        param = param_dict[target]
                        weight_loader = param.weight_loader
                        weight_loader(param, value, shard_id)
                        stacked = True
                        break

                if not stacked:
                    raise ValueError(f"Unknown parameter {name}")
            else:
                raise ValueError(f"Unknown parameter {name}")

            del value

        return _eval_model(
            self.model,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            task_manager=self.task_manager,
            batch_size=self.batch_size,
            apply_chat_template=self.config.apply_chat_template,
            fewshot_as_multiturn=self.config.fewshot_as_multiturn,
        )

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> dict:
        return self.evaluate(genotype)

    def smoke_test_model(self, model, task_manager=None):
        """Quick test to check if model produces coherent responses."""
        try:
            # Simple smoke test prompt
            test_prompt = "Tell me about yourself."

            # Generate response
            if hasattr(model, 'generate'):
                response = model.generate(test_prompt, max_tokens=4096)
            else:
                # Fallback for different model types
                response = "Model response test failed"

            print(f"Current Merged' model's response is {response[:1000]}")

            # Check for rambling / repetition
            if self.detect_rambling_simple(response):
                return False

            return True
        except Exception as e:
            logging.warning(f"Smoke test failed: {e}")
            return False

    def detect_rambling_simple(self, text):
        """Simple detection of rambling/repetition."""
        # Check for excessive repetition
        words = text.split()
        rambling_cnt = 0
        if len(words) > 20:
            # Count repeated phrases
            for i in range(len(words) - 5):
                phrase = ' '.join(words[i:i+5])
                if text.count(phrase) > 2:
                    rambling_cnt += 1
                    if rambling_cnt > 3:
                        return True
        return False