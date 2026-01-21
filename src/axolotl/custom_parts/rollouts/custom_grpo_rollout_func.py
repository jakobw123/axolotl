from collections.abc import Callable
from typing import Any, List, Dict, TypedDict, Union
from copy import deepcopy

from axolotl.custom_parts.code_interpreter.parser import extract_jupyter_like_program
from axolotl.custom_parts.code_interpreter.python_executor import PythonExecutor
from pydantic import BaseModel, ConfigDict
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.import_utils import is_vllm_available
from trl.extras.profiling import profiling_context
from trl.extras.vllm_client import VLLMClient
from vllm import TokensPrompt, LLM

from axolotl.core.trainers.grpo import AxolotlGRPOTrainer, AxolotlGRPOSequenceParallelTrainer
from axolotl.custom_parts.utils import RoleConfig, RolloutReturn

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


def interleaved_rollout_func(
    prompts: List[str], 
    trainer: Any
) -> Dict[str, Any]:
    """
    Handles multi-turn interleaved code generation + execution.
    Returns flattened completion_ids and the corresponding completion_mask.
    """
    
    mode = "train" if trainer.model.training else "eval"
    num_generations = trainer.num_generations if mode == "train" else trainer.num_generations_eval
    tokenizer = trainer.processing_class
    
    # Expand unique prompts: [P1, P2] -> [P1_1, P1_2, P2_1, P2_2]
    # We work with strings for vLLM input, but accumulate IDs for output
    current_prompts = [p for p in prompts for _ in range(num_generations)]
    batch_size = len(current_prompts)
    
    initial_prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in current_prompts_text]
    accumulated_tokens = [len(ids) for ids in initial_prompt_ids]
    
    # These buffers hold ONLY the generated/executed part (not the input prompt)
    # This is what will be returned to the trainer
    completion_ids = [[] for _ in range(batch_size)]
    completion_mask = [[] for _ in range(batch_size)]
    is_completed = [False] * batch_size
    per_example_exec_count = [0] * batch_size
    
    
    output_roles = getattr(trainer.args, "output_roles", {})
    
    reasoning_config = output_roles.get("reasoning")
    think_start_tag = reasoning_config.start_tag if reasoning_config else "<think>"
    has_thinking = [(f"{think_start_tag}\n" if ip.strip().endswith(think_start_tag) else "") for ip in prompts]
    has_any_thinking = any(has_thinking)
    
    code_config = output_roles.get("code_block", {})
    res_config = output_roles.get("code_result", {})
    
    code_start_tag = code_config.get("start_tag", "<code_block>")
    code_end_tag = code_config.get("end_tag", "</code_block>")
    res_start_tag = res_config.get("start_tag", "<code_result>")
    res_end_tag = res_config.get("end_tag", "</code_result>")
    
    max_turns = getattr(trainer.args, "max_turns", 5)
    
    base_stop = trainer.args.generation_kwargs.get("stop", [])
    if code_end_tag not in base_stop:
        base_stop = list(base_stop) + [code_end_tag]
        
    for turn in range(max_turns):

        active_indices = [i for i, done in enumerate(is_completed) if not done]
        if not active_indices:
            break
            
        active_prompts = [current_prompts[i] for i in active_indices]
        
        valid_batch_ids = []
        valid_indices_map = [] # Maps index in valid_batch -> index in active_indices

        safety_limit = trainer.vllm_mode .max_model_len
        max_new = trainer.args.get("max_completion_length")

        for i, seq_prompt in enumerate(active_prompts):
            current_len = len(seq_ids)
            
            if current_len + max_new > safety_limit:
                original_idx = active_indices[i]
                
                print(f"Example {original_idx} exceeded context limit ({current_len} + {max_new} > {safety_limit}). Stopping.")
                
                is_completed[original_idx] = True
                final_outputs[original_idx] += "\n[SYSTEM: Context Window Exceeded - Stopped]"
            else:
                valid_batch_ids.append(TokensPrompt(prompt_token_ids=seq_ids))
                valid_indices_map.append(i)
                
        if not valid_batch_ids:
            continue
        
        # --- A. CALL vLLM ---
        # We handle both Server and Colocate modes
        outputs = _generate_vllm_batch(
            prompts=active_prompts, 
            trainer=trainer, 
            additional_sampling_parameters={
                "stop": base_stop,
                "include_stop_str": True
            }
        )
        
        code_snippets = []
        indices_with_code = []
        
        for i, output in enumerate(outputs):
            idx_in_active = valid_indices_map[i]
            original_idx = active_indices[idx_in_active]
            
            # global_idx = active_indices[idx_in_batch]
            
            generated_ids = output.token_ids
            generated_text = output.text
            finish_reason = output.finish_reason
            
            completion_ids[global_idx].extend(generated_ids)
            completion_mask[global_idx].extend([1] * len(generated_ids))
            
            current_prompts[global_idx] += generated_text
            
            if finish_reason == "stop" and generated_text.rstrip().endswith(code_end_tag):
                code = extract_jupyter_like_program(
                    final_outputs[original_idx], 
                    start_tag=code_start_tag,
                    end_tag=code_end_tag,
                    res_start_tag=res_start_tag,
                    res_end_tag=res_end_tag
                )
                code_snippets.append(code)
                indices_with_code.append(original_idx)
                per_example_exec_count[original_idx] += 1
                # Extract code for execution
                # Simple extraction strategy: take everything between last tags
                try:
                    # Find the last occurrence of start tag
                    start_idx = generated_text.rfind(code_start_tag)
                    if start_idx != -1:
                        # Strip tags to get clean code
                        raw_code = generated_text[start_idx + len(code_start_tag) : -len(code_end_tag)]
                        exec_queue.append((global_idx, raw_code))
                    else:
                        # Malformed generation (end tag without start tag)
                        # Treat as finished or just continue? Let's finish to be safe.
                        is_completed[global_idx] = True
                except Exception:
                    is_completed[global_idx] = True
            else:
                is_completed[global_idx] = True

        if exec_queue:
            global_indices = [x[0] for x in exec_queue]
            code_snippets = [x[1] for x in exec_queue]
        
            exec_results = trainer.code_executor.batch_apply(code_snippets)
            
            for global_idx, (result, error) in zip(global_indices, exec_results):
                # Format Output
                content = error if error else result
                # Truncate if tool output is massive to prevent OOM
                if len(content) > 10000: 
                    content = content[:10000] + "...[Truncated]"
                    
                formatted_output = f"\n{res_start_tag}\n{content}\n{res_end_tag}\n"
                
                # Tokenize Tool Output
                # add_special_tokens=False is CRITICAL here (don't want extra BOS/EOS in middle)
                tool_ids = tokenizer.encode(formatted_output, add_special_tokens=False)
                
                # 5. Append to Buffers (Tool Generated -> Mask = 0)
                completion_ids[global_idx].extend(tool_ids)
                completion_mask[global_idx].extend([0] * len(tool_ids))
                
                # 6. Update Prompt State
                current_prompts[global_idx] += formatted_output
                
                # Ensure this sequence stays active for next turn
                is_completed[global_idx] = False

    # ------------------------------------------------------------------
    # 4. FINALIZE & RETURN
    # ------------------------------------------------------------------
    return {
        "completion_ids": completion_ids,   # List[List[int]]
        "completion_mask": completion_mask  # List[List[int]] (0s and 1s)
    }


def _run_interleaved_mode(
    prompts: List[str],
    trainer: Union[AxolotlGRPOTrainer, AxolotlGRPOSequenceParallelTrainer, GRPOTrainer],
    code_executor: PythonExecutor,
    max_turns: int = 5,
    timeout: float = 50,
    local_num_procs: int = 1,
) -> RolloutReturn:
    
    
    initial_prompts = deepcopy(prompts)
    
    output_roles: Dict[str, RoleConfig] = getattr(trainer.args, "output_roles", {})
    
    reasoning_config = output_roles.get("reasoning")
    think_start_tag = reasoning_config.start_tag if reasoning_config else "<think>"
    has_thinking = [(f"{think_start_tag}\n" if ip.strip().endswith(think_start_tag) else "") for ip in initial_prompts]
    has_any_thinking = any(has_thinking)
    
    code_config = output_roles.get("code_block")
    res_config = output_roles.get("code_result")
    final_config = output_roles.get("final_answer")
    code_start_tag = code_config.start_tag if code_config else "<code_block>"
    code_end_tag = code_config.end_tag if code_config else "</code_block>"
    res_start_tag = res_config.start_tag if res_config else "<code_result>"
    res_end_tag = res_config.end_tag if res_config else "</code_result>"
    final_start_tag = final_config.start_tag if final_config else "<final_answer>"
    final_end_tag = final_config.end_tag if final_config else "</final_answer>"

    per_example_exec_count = [0] * len(prompts)
    is_completed = [False] * len(prompts)
    accumulated_tokens = [0] * len(prompts)
    final_outputs = [""] * len(prompts)
    code_block_success = [[] for _ in range(len(prompts))]
    
    global_loop_limit = max_turns * 5 
    current_global_loop = 0

    while True:
        current_global_loop += 1
        print(f"--- Interleaved Step {current_global_loop} (Global) ---")
        
        active_indices = []
        for i, done in enumerate(is_completed):
            if not done:
                if per_example_exec_count[i] >= max_turns:
                    is_completed[i] = True
                else:
                    active_indices.append(i)
        
        if not active_indices or current_global_loop > global_loop_limit:
            if current_global_loop > global_loop_limit:
                print("Warning: Hit global loop safety limit. Stopping.")
            break
        
        active_inputs = [prompts[i] for i in active_indices]
        
        raw_batch_ids = tokenizer_strategy.tokenizer.apply_chat_template(
            active_inputs,
            tokenize=True,
            padding=False,
            truncation=False,
            add_generation_prompt=True 
        )
        # batch_ids = []
        # raw_batch_ids = tokenizer_strategy.tokenizer.apply_chat_template(
        #     active_inputs,
        #     tokenize=True,
        #     padding=False,
        #     truncation=True,
        #     max_length=shared_config.max_model_len - sampling_params.max_tokens
        # )
        
        # for single_seq_ids in raw_batch_ids:
        #     batch_ids.append(TokensPrompt(prompt_token_ids=single_seq_ids))
        valid_batch_ids = []
        valid_indices_map = [] # Maps index in valid_batch -> index in active_indices

        safety_limit = trainer.vllm_mode .max_model_len
        max_new = sampling_params.max_tokens

        for i, seq_ids in enumerate(raw_batch_ids):
            current_len = len(seq_ids)
            
            if current_len + max_new > safety_limit:
                original_idx = active_indices[i]
                
                print(f"Example {original_idx} exceeded context limit ({current_len} + {max_new} > {safety_limit}). Stopping.")
                
                is_completed[original_idx] = True
                final_outputs[original_idx] += "\n[SYSTEM: Context Window Exceeded - Stopped]"
            else:
                valid_batch_ids.append(TokensPrompt(prompt_token_ids=seq_ids))
                valid_indices_map.append(i)
                
        if not valid_batch_ids:
            continue
                
        outputs = model.generate(
            valid_batch_ids,
            sampling_params=sampling_params
        )
        code_snippets = []
        indices_with_code = []
        
        for i, output in enumerate(outputs):
            idx_in_active = valid_indices_map[i]
            original_idx = active_indices[idx_in_active]
            
            generated_text = output.outputs[0].text
            
            batch_messages[original_idx].append({
                "role": "assistant", 
                "content": generated_text
            })
            
            final_outputs[original_idx] += (f"{think_start_tag}\n" + generated_text if has_any_thinking else generated_text)
            accumulated_tokens[original_idx] += len(output.outputs[0].token_ids)
            
            finish_reason = output.outputs[0].finish_reason
            hit_custom_stop = (finish_reason == "stop" and generated_text.strip().endswith(code_end_tag))
            
            if hit_custom_stop:
                code = extract_jupyter_like_program(
                    final_outputs[original_idx], 
                    start_tag=code_start_tag,
                    end_tag=code_end_tag,
                    res_start_tag=res_start_tag,
                    res_end_tag=res_end_tag
                )
                code_snippets.append(code)
                indices_with_code.append(original_idx)
                per_example_exec_count[original_idx] += 1
                
            else:
                is_completed[original_idx] = True
        
        if code_snippets:
            print(f"Executing {len(code_snippets)} blocks...")
            results = code_executor.batch_apply(code_snippets)
            
            for original_idx, (res, report) in zip(indices_with_code, results):
                if len(report):
                    output_content = report
                    code_block_success[original_idx].append(False)
                    
                else:
                    output_content = res
                    code_block_success[original_idx].append(True)
                
                formatted_result = f"{res_start_tag}\n{output_content}\n{res_end_tag}"
                
                batch_messages[original_idx].append({
                    "role": "user", 
                    "content": formatted_result
                })
                
                final_outputs[original_idx] += f"\n{formatted_result}\n"

    
    final_outputs = [(ht + rp if not rp.lstrip().startswith(think_start_tag) else rp) for ht, rp in zip(has_thinking, final_outputs)]
    
    exp_result = _pack_results(
        normalized_examples=batch_normalized_examples,
        prompts=initial_prompts,
        raw_predictions=final_outputs,
        lengths=accumulated_tokens,
        time_taken=inference_time
    )
    
    for j, c in enumerate(code_block_success):
        exp_result.all_per_example_results.metrics[j]["code_block_success"] = c
        
    return exp_result

def _generate_vllm_batch(
    prompts: List[str],
    trainer: Union[AxolotlGRPOTrainer, AxolotlGRPOSequenceParallelTrainer, GRPOTrainer],
    additional_sampling_parameters: Dict[str, Any] = None
):
    
    mode = "train" if trainer.model.training else "eval"
    num_generations = trainer.num_generations if mode == "train" else trainer.num_generations_eval
    
    if hasattr(trainer, "llm") and trainer.llm is not None:
        
    elif hasattr(trainer, "vllm_client") and trainer.vllm_client is not None:
        
    else:
        raise ValueError("No vLLM engine found in trainer")

def rollout_func_with_code_interleaved(
    prompts: List[str], 
    trainer: Union[AxolotlGRPOTrainer, AxolotlGRPOSequenceParallelTrainer, GRPOTrainer]
) -> RolloutReturn:
    """
    Args:
        prompts: A list of unique prompts. (Axolotl has already removed duplicates)
        trainer: The trainer instance.
        
    Returns:
        dict: Must contain key "completion_ids".
              Value must be a list of lists of integers (token IDs).
              Size: len(prompts) * num_generations
    """
    
    mode = "train" if trainer.model.training else "eval"
    num_generations = trainer.num_generations if mode == "train" else trainer.num_generations_eval
    
    if not prompts:
        return RolloutReturn(completion_ids=[], prompt_ids=[], logprobs=[])
    
    if not trainer.use_vllm: 
        return RolloutReturn(completion_ids=[], prompt_ids=[], logprobs=[])
    
    output_roles: Dict[str, RoleConfig] = getattr(trainer.args, "output_roles", {})
    
    if not output_roles:
        return RolloutReturn(completion_ids=[], prompt_ids=[], logprobs=[])
    
    code_config = output_roles.get("code_block") 
    code_start_tag = code_config.start_tag if code_config else "<code_block>"
    code_end_tag = code_config.end_tag if code_config else "</code_block>"
    
    model = None
    sampling_params = None
    if trainer.vllm_mode == "server":
        model = trainer.vllm_client

        sampling_params = {
            "n": num_generations,
            "repetition_penalty": trainer.repetition_penalty,
            "temperature": trainer.temperature,
            "top_p": trainer.top_p,
            "top_k": -1 if trainer.top_k is None else trainer.top_k,
            "min_p": 0.0 if trainer.min_p is None else trainer.min_p,
            "max_tokens": trainer.max_completion_length,
            "guided_decoding_regex": trainer.guided_decoding_regex,
            "generation_kwargs": trainer.args.generation_kwargs,
        }
        
        if not sampling_params["generation_kwargs"]:
            sampling_params["generation_kwargs"] = {}
            
        original_stop = sampling_params["generation_kwargs"].get("stop") or []
        if code_end_tag not in original_stop:
            sampling_params["generation_kwargs"]["stop"] = original_stop + [code_end_tag]
        sampling_params["generation_kwargs"]["include_stop_str_in_output"] = True 
        
    else:
        model = trainer.llm
        
        if trainer.guided_decoding_regex:
            guided_decoding = GuidedDecodingParams(regex=trainer.guided_decoding_regex)
            
        else:
            guided_decoding = None
            
        generation_kwargs = {
            "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
            "repetition_penalty": trainer.repetition_penalty,
            "temperature": trainer.temperature,
            "top_p": trainer.top_p,
            "top_k": -1 if trainer.top_k is None else trainer.top_k,
            "min_p": 0.0 if trainer.min_p is None else trainer.min_p,
            "max_tokens": trainer.max_completion_length,
            "guided_decoding": guided_decoding,
            "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
        }
        
        if trainer.args.generation_kwargs is not None:
            generation_kwargs.update(trainer.args.generation_kwargs)
            
        sampling_params = SamplingParams(**generation_kwargs)
          
        original_stop = sampling_params.stop or []
        if code_end_tag not in original_stop:
            sampling_params.stop = original_stop + [code_end_tag]
        sampling_params.include_stop_str_in_output = True 
    
    
    if not model or not sampling_params:
        return RolloutReturn(completion_ids=[], prompt_ids=[], logprobs=[])
    
    rollout_return = _run_interleaved_mode(prompts, trainer, code_executor, )
    
    return rollout_return.model_dump()
    

    
    
