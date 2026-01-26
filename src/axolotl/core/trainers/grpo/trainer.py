"""Axolotl GRPO trainers (with and without sequence parallelism handling)"""

from collections.abc import Callable
import copy
import warnings
from functools import partial
from typing import Any
from packaging.version import Version

import transformers
from .......src.code_interpreter.python_executor import PythonExecutor
from .......src.output_handler.parsing.base import Parser
from .......src.output_handler.references.base import ReferenceBuilder
import datasets
import torch
import torch.distributed as dist
import torch.utils.data
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_available
)
from datasets import Dataset, IterableDataset
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
)
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from trl import GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
    prepare_multimodal_messages
)
from trl.chat_template_utils import parse_response
from trl.import_utils import is_vllm_available
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import RewardFunc, nanstd, RolloutFunc
from trl.trainer.utils import pad, nanmax, nanmin

from axolotl.core.trainers.grpo.sampler import SequenceParallelRepeatRandomSampler
from axolotl.core.trainers.mixins import (
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.monkeypatch.ring_attn import get_ring_attn_group

if is_peft_available():
    from peft import PeftConfig
    
if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


class AxolotlGRPOTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    GRPOTrainer,
):
    """Extend the base GRPOTrainer for axolotl helpers"""

    _tag_names = ["trl", "grpo", "axolotl"]
    
    def __init(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: (
            Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None
        ) = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        reward_processing_classes: (
            PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None
        ) = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None
    ):
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            tools=tools,
            rollout_func=rollout_func
        )
        
        # if not self.tools:
        #     self.tools = ["dummy_tool_for_masking"]
        
        self.code_executor: PythonExecutor = None
        if self.args.use_code_executor and self.args.sandbox_python_path and self.args.sandbox_script_path:
            
            self.code_executor = PythonExecutor(
                sandbox_python_path=self.args.sandbox_python_path,
                server_script_path=self.args.sandbox_script_path,
                get_answer_from_stdout=True,
                # hard_timeout_padding=timeout,
                local_num_procs=self.args.local_num_procs if self.args.local_num_procs else 1
            )
        
        output_roles = getattr(self.args, "output_roles", {})
        self.parser = Parser(output_roles)
        
        active_engine = getattr(self, "vllm_client", None) or getattr(self, "llm", None)
        
        for reward_fn in self.reward_funcs:
            # Check if this reward function is our SelfReward type
            if active_engine and hasattr(reward_fn, "set_vllm_engine"):
                reward_fn.set_vllm_engine(active_engine)

            if hasattr(reward_fn, "set_context"):
                reward_fn.set_context(self.parser)
                
                
                
    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        prompt_ids, completion_ids, logprobs, extra_fields = self._generate_single_turn(prompts)

        # Decode completions. It's important to use `parse_response` when possible, because it handles tool calls.
        if is_conversational({"prompt": prompts[0]}):
            if (
                Version(transformers.__version__) >= Version("5.0.0.dev0")  # parse_response added in v5
                and isinstance(self.processing_class, PreTrainedTokenizerBase)  # doesn't work with processors
                and hasattr(self.processing_class, "response_schema")  # attribute not set by default for now
                and self.processing_class.response_schema is not None  # only works if the tokenizer has a schema
            ):
                completions = [[parse_response(self.processing_class, ids)] for ids in completion_ids]
            else:
                contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [[{"role": "assistant", "content": content}] for content in contents]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Extract tool calls from the completions and (possibly) execute them
        if self.tools:
            (
                tool_mask,
                completions,
                completion_ids,
                logprobs,
                tool_call_count,
                tool_failure_count,
            ) = self._tool_call_loop(prompts, prompt_ids, completion_ids, completions, logprobs)
            
        elif self.code_executor:
            tool_mask = extra_fields["code_result_mask"]
            code_call_count = extra_fields["code_call_count"]
            code_failure_count = extra_fields["code_failure_count"]
            
        else:
            tool_mask = None

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:  # count only non-tool tokens (tool_mask=1)
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        if self.tools:
            agg_tool_call_count = self.accelerator.gather(torch.tensor(tool_call_count, device=device)).sum()
            tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
            self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
            agg_tool_failure_count = self.accelerator.gather(torch.tensor(tool_failure_count, device=device)).sum()
            failure_frequency = (
                (agg_tool_failure_count / agg_tool_call_count).item() if agg_tool_call_count > 0 else 0.0
            )
            self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)
            
        if self.code_executor:
            agg_code_call_count = self.accelerator.gather(torch.tensor(code_call_count, device=device)).sum()
            tool_call_frequency = (agg_code_call_count / len(agg_prompt_lengths)).item()
            self._metrics[mode]["code/call_frequency"].append(tool_call_frequency)
            agg_code_failure_count = self.accelerator.gather(torch.tensor(code_failure_count, device=device)).sum()
            failure_frequency = (
                (agg_code_failure_count / agg_code_call_count).item() if agg_code_call_count > 0 else 0.0
            )
            self._metrics[mode]["code/failure_frequency"].append(failure_frequency)

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )
                
    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if self.tools or self.code_executor:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")  # 0 for tool result tokens, 1 elsewhere

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask * tool_mask if self.tools or self.code_executor else completion_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = (completion_mask * tool_mask).bool() if self.tools or self.code_executor else completion_mask.bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if self.tools or self.code_executor:
            output["tool_mask"] = tool_mask
        return output
    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            mask = completion_mask * inputs["tool_mask"] if self.tools or self.code_executor else completion_mask
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            mask = completion_mask * inputs["tool_mask"] if self.tools or self.code_executor else completion_mask
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mask = completion_mask * inputs["tool_mask"] if self.tools or self.code_executor else completion_mask
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss
                
class AxolotlGRPOSequenceParallelTrainer(AxolotlGRPOTrainer):
    """Extend the base GRPOTrainer for sequence parallelism handling"""

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: (
            Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None
        ) = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        reward_processing_classes: (
            PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None
        ) = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = (None, None),
        peft_config: "PeftConfig | None" = None,
        optimizer_cls_and_kwargs: tuple[type, dict] | None = None,
        rollout_func: RolloutFunc | None = None
    ):
        # First call the superclass constructor with all arguments
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            rollout_func=rollout_func
        )

        # Get number of SP groups (number of processes divided by SP degree)
        num_processes = self.accelerator.num_processes
        num_sp_groups = num_processes // self.args.context_parallel_size

        # Calculate batch size per SP group (not per process)
        sp_group_batch_size = self.args.per_device_train_batch_size * num_sp_groups
        possible_values = [
            n_gen
            for n_gen in range(2, sp_group_batch_size + 1)
            if (sp_group_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The batch size per SP group ({num_sp_groups} x "
                f"{self.args.per_device_train_batch_size}) must be evenly divisible by "
                f"the number of generations per prompt ({self.num_generations}). Given "
                "the current configuration, the valid values for the number of "
                f"generations are: {possible_values}."
            )

        if self.args.eval_strategy != "no":
            # If sequence parallelism is enabled, calculate batch size per SP group
            sp_group_eval_batch_size = args.per_device_eval_batch_size * num_sp_groups  # type: ignore[union-attr]
            possible_values = [
                n_gen
                for n_gen in range(2, sp_group_eval_batch_size + 1)
                if (sp_group_eval_batch_size) % n_gen == 0
            ]

            if self.num_generations not in possible_values:
                raise ValueError(
                    f"With sequence parallelism (degree {self.args.context_parallel_size}), "
                    f"the eval batch size per SP group ({num_sp_groups} x {self.args.per_device_eval_batch_size}) "
                    f"must be evenly divisible by the number of generations per prompt "
                    f"({self.num_generations}). Given the current eval batch size, "
                    f"the valid values for the number of generations are: {possible_values}."
                )

        self.sp_group = None
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = 0
        self.local_world_size = 1

    def train(self, *args, **kwargs):
        # Initialize the SP group
        self.sp_group = get_ring_attn_group()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank(group=self.sp_group)
        self.local_world_size = dist.get_world_size(group=self.sp_group)

        return super().train(*args, **kwargs)

    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.world_size
            * self.args.gradient_accumulation_steps
        )

        return SequenceParallelRepeatRandomSampler(
            dataset=self.train_dataset,
            mini_repeat_count=self.num_generations,
            world_size=self.world_size,
            rank=self.rank,
            batch_size=effective_batch_size
            // self.num_generations
            // self.args.context_parallel_size,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            context_parallel_size=self.args.context_parallel_size,
            shuffle=True,
            seed=self.args.seed,
            drop_last=True,
        )

    def _create_dataloader_params(self, is_eval=False, custom_batch_size=None):
        """Create common dataloader parameters for train or eval."""
        batch_size = custom_batch_size or (
            self.args.eval_batch_size if is_eval else self._train_batch_size
        )

        params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        # Add persistent workers only for training
        if not is_eval and hasattr(self.args, "dataloader_persistent_workers"):
            params["persistent_workers"] = self.args.dataloader_persistent_workers

        # Add prefetch factor if specified
        if self.args.dataloader_prefetch_factor:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return params

    def _prepare_dataloader(
        self, dataset, sampler, is_eval=False, custom_batch_size=None
    ):
        """Prepare a dataloader with the given dataset and sampler."""
        # Get base parameters
        dataloader_params = self._create_dataloader_params(is_eval, custom_batch_size)

        # Add sampler configuration
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if isinstance(sampler, BatchSampler):
                # batch_size and batch_sampler are mutually exclusive
                dataloader_params["batch_sampler"] = sampler
                del dataloader_params["batch_size"]
            else:
                dataloader_params["sampler"] = sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last

            if not is_eval:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )

        # Create the dataloader
        dataloader = DataLoader(dataset, **dataloader_params)

        if self.args.sample_packing and (
            (not is_eval and not self.args.pretraining)
            or (is_eval and self.args.eval_sample_packing is not False)
        ):
            self.accelerator.even_batches = False

        # Return unprepared dataloader if using sequence parallelism
        # TODO(djsaunde): We might be able to use `accelerate`'s dataloader preparation
        # if we use `dispatch_batches` and `slice_fn_for_dispatch` properly (i.e.,
        # slice each batch along the sequence dimension).
        if self.args.context_parallel_size > 1:
            return dataloader

        # Otherwise prepare with accelerator
        return self.accelerator.prepare_data_loader(dataloader)

    def get_train_dataloader(self) -> DataLoader:
        """Get dataloader for training"""
        train_dataset = self.train_dataset

        data_collator = self.data_collator  # type: ignore

        # Handle dataset preprocessing
        if isinstance(train_dataset, datasets.Dataset):
            # Add debug print before any modifications
            if self.args.sample_packing and not self.args.pretraining:
                train_dataset = train_dataset.remove_columns(["length"])
            if not self.args.sample_packing or self.args.pretraining:
                train_dataset = self._remove_unused_columns(
                    train_dataset, description="training"
                )
        else:
            self.data_collator = self._get_collator_with_removed_columns(
                data_collator,
                description="training",
            )

        # Get sampler and create dataloader
        sampler = self._get_train_sampler()
        dataloader = self._prepare_dataloader(train_dataset, sampler, is_eval=False)

        return dataloader

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            custom_mask_list = None

            if self.state.global_step != self._last_loaded_step:  # type: ignore[has-type]
                self._move_model_to_vllm()

                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                if self.args.context_parallel_size > 1:
                    # Calculate sequence parallel group information
                    world_size = self.accelerator.num_processes
                    context_parallel_size = self.args.context_parallel_size
                    num_sp_groups = world_size // context_parallel_size

                    # Since processes in the same SP group have the same prompts, we need to ensure
                    # we only take one copy of each prompt from each SP group
                    ordered_set_of_prompts = []
                    for sp_group_id in range(num_sp_groups):
                        # Get the first process from each SP group (typically the group leader)
                        group_leader_rank = sp_group_id * context_parallel_size

                        # Extract prompts from this SP group, accounting for num_generations duplicates
                        # We only need prompts from one rank in each SP group
                        group_prompts = all_prompts_text[
                            group_leader_rank * len(prompts_text) : (
                                group_leader_rank + 1
                            )
                            * len(prompts_text) : self.num_generations
                        ]

                        ordered_set_of_prompts.extend(group_prompts)
                else:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[
                        :: self.num_generations * self.args.context_parallel_size
                    ]
                    
                if self.rollout_func is not None:
                    # Use custom rollout (handles multi-turn/tools)
                    # We pass the deduped prompts prepared by Axolotl above
                    # rollout_prompts = ordered_set_of_prompts
                    # if rollout_prompts and is_conversational({"prompt": rollout_prompts[0]}):
                    #     rollout_prompts = [
                    #         apply_chat_template(
                    #             {"prompt": p}, self.processing_class, **self.chat_template_kwargs
                    #         )["prompt"]
                    #         for p in rollout_prompts
                    #     ]
                    with profiling_context(self, "vLLM.generate"):
                        rollout_output = self.rollout_func(ordered_set_of_prompts, self)
                    
                    completion_ids = rollout_output.get("completion_ids", [])
                    custom_mask_list = rollout_output.get("completion_mask", None)
                    
                elif hasattr(self, "llm") and self.llm is not None:
                    if self.guided_decoding_regex:
                        guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                        
                    else:
                        guided_decoding = None

                    generation_kwargs = {
                        "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": -1 if self.top_k is None else self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": self.max_completion_length,
                        "guided_decoding": guided_decoding,
                        "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
                    }
                    
                    if self.args.generation_kwargs is not None:
                        generation_kwargs.update(self.args.generation_kwargs)
                        
                    sampling_params = SamplingParams(**generation_kwargs)
                    
                    with profiling_context(self, "vLLM.generate"):
                        req_outputs = self.llm.generate(ordered_set_of_prompts, sampling_params=sampling_params, use_tqdm=False)
                    
                    # Flatten results: vLLM returns [ReqOutput(outputs=[Comp1, Comp2]), ...]
                    # Flat list: [Comp1_P1, Comp2_P1, Comp1_P2, Comp2_P2, ...]
                    completion_ids = [output.token_ids for req in req_outputs for output in req.outputs]
                    
                else:
                    num_generations = self.num_generations if mode == "train" else self.num_generations_eval
                    
                    sampling_params = {
                        "n": num_generations,
                        "repetition_penalty": self.repetition_penalty,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "top_k": -1 if self.top_k is None else self.top_k,
                        "min_p": 0.0 if self.min_p is None else self.min_p,
                        "max_tokens": self.max_completion_length,
                        "guided_decoding_regex": self.guided_decoding_regex,
                        "generation_kwargs": self.args.generation_kwargs,
                    }

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            **sampling_params
                        )
                        
                        if isinstance(output, dict) and "completion_ids" in output:
                            completion_ids = output["completion_ids"]
                        else:
                            completion_ids = output
            else:
                completion_ids = [None] * (
                    len(all_prompts_text) // self.args.context_parallel_size
                )
                custom_mask_list = [None] * (
                    len(all_prompts_text) // self.args.context_parallel_size
                )

            # Broadcast the completions from the main process to all processes
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            
            has_mask = self.accelerator.gather(torch.tensor([1 if custom_mask_list is not None else 0], device=device)).sum() > 0
            if has_mask:
                custom_mask_list = broadcast_object_list(custom_mask_list, from_process=0)

            # Determine the appropriate slice based on sequence parallelism
            if self.args.context_parallel_size > 1:
                # Calculate SP group ID (which group of ranks this rank belongs to)
                sp_group_id = self.accelerator.process_index // self.local_world_size

                # Calculate the start index for this SP group
                sp_group_start = sp_group_id * len(prompts) * self.local_world_size

                # All ranks in the same SP group get the same data slice
                process_slice = slice(
                    sp_group_start,
                    sp_group_start + len(prompts),
                )
                
                completion_ids = completion_ids[process_slice]
                    
                if custom_mask_list:
                    custom_mask_list = custom_mask_list[process_slice]

            else:
                # Original behavior for non-sequence parallel case
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
            
                completion_ids = completion_ids[process_slice]
                    
                if custom_mask_list:
                    custom_mask_list = custom_mask_list[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            
            if custom_mask_list is not None:
                completion_mask = [torch.tensor(m, device=device) for m in custom_mask_list]
                completion_mask = pad(completion_mask, padding_value=0) # Pad with 0 (ignore)
                
            else:
                completion_mask = [torch.ones_like(ids) for ids in completion_ids]
                completion_mask = pad(completion_mask, padding_value=0)
            
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.args.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=False):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
                strict=False,
            )
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c}
                            for p, c in zip(prompts, completions, strict=False)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [
                            p + c for p, c in zip(prompts, completions, strict=False)
                        ]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [
                        key for key in inputs[0] if key not in ["prompt", "completion"]
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs] for key in keys
                    }
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward.",
                stacklevel=2,
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        if self.args.context_parallel_size > 1:
            # Calculate SP group ID (which group of ranks this rank belongs to)
            sp_group_id = self.accelerator.process_index // self.local_world_size

            # Calculate the start index for this SP group
            sp_group_start = sp_group_id * len(prompts) * self.local_world_size

            # All ranks in the same SP group get the same data slice
            process_slice = slice(
                sp_group_start,
                sp_group_start + len(prompts),
            )
        else:
            # Original behavior for non-sequence parallel case
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self._total_train_tokens += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        )
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_mask.float().max().item()
        )

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(
            agg_completion_mask
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_mask.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
