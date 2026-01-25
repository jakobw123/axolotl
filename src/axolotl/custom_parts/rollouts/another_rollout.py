import torch
import copy

def grpo_rollout(prompts, trainer):
    """
    Custom rollout function for interleaved reasoning and code execution.
    
    Args:
        prompts: List of prompt messages (list of dicts) or strings. 
                 If strings, they are converted to messages.
        trainer: The GRPO trainer instance.
        
    Returns:
        dict containing 'prompt_ids', 'completion_ids', 'completion_mask'
    """
    tokenizer = trainer.processing_class
    
    # Ensure prompts are in message format
    if isinstance(prompts[0], str):
        # Fallback if raw strings passed, though generally we expect messages
        current_prompts_msgs = [[{"role": "user", "content": p}] for p in prompts]
    else:
        current_prompts_msgs = copy.deepcopy(prompts)
        
    batch_size = len(current_prompts_msgs)
    
    # Configuration for interleaved generation
    # You might want to pull these from trainer.args or a custom config object attached to trainer
    max_turns = 5 
    code_start_tag = "<code_block>"
    code_end_tag = "</code_block>"
    res_start_tag = "<code_result>"
    res_end_tag = "</code_result>"
    stop_tokens = [code_end_tag, tokenizer.eos_token] 
    
    # Track completion IDs and Mask for every sample
    # We will accumulate tokens here. 
    # Logic: 
    # 1. Start with empty completion_ids list for each sample.
    # 2. Each turn, generate. Append IDs to completion_ids. Append 1s to mask.
    # 3. If code generated, execute. Append result IDs to completion_ids. Append 0s to mask.
    
    all_completion_ids = [[] for _ in range(batch_size)]
    all_completion_masks = [[] for _ in range(batch_size)]
    active_indices = list(range(batch_size))
    
    # We need the initial prompt IDs to calculate the starting point for completion
    # But since TRL splits prompt/completion logic, we only return completion parts.
    # However, for vLLM generation we need the full prompt history.
    
    for turn in range(max_turns):
        if not active_indices:
            break
            
        # Prepare inputs for vLLM
        # We apply chat template to the CURRENT history of messages
        vllm_prompts = []
        for i in active_indices:
            # apply_chat_template returns string. vLLM handles tokenization if we pass strings?
            # Or we pass token_ids? Passing strings is safer for template logic.
            # We assume trainer.vllm_client or trainer.llm (colocate) is used.
            # If using custom executor in Axolotl trainer, we might need to access the client differently.
            # Since this function runs inside the trainer methods we added:
            
            # Note: We must ensure generation prompt is added to force assistant response
            prompt_str = tokenizer.apply_chat_template(
                current_prompts_msgs[i], 
                tokenize=False, 
                add_generation_prompt=True
            )
            vllm_prompts.append(prompt_str)
            
        # Generate
        # We need to access the vLLM client from the trainer
        # NOTE: This assumes standard Axolotl vLLM setup
        sampling_params = {
            "n": 1, # We manage generations manually (1 per prompt in this loop, but prompts might be duplicated upstream)
            "temperature": trainer.temperature,
            "max_tokens": trainer.max_completion_length, # Should handle remaining length dynamically?
            "stop": stop_tokens,
            # ... other params from trainer ...
        }
        
        # Call vLLM
        if trainer.vllm_mode == "server":
            # Server mode returns list of list of IDs
            outputs = trainer.vllm_client.generate(vllm_prompts, **sampling_params)
        else:
            # Colocate mode
            # We construct SamplingParams object
            from vllm import SamplingParams
            sp = SamplingParams(**sampling_params)
            # Use trainer.llm.generate
            # Note: prompts might need to be broadcast in colocate if not careful, 
            # but since we are running this ON THE MAIN PROCESS inside the trainer hook we added,
            # we should treat it like server mode logic or ensure colocate logic works.
            # Actually, standard GRPO colocate runs generation on all ranks.
            # If we run this on main process only (as per my trainer modification), 
            # we can't use colocate easily unless we are careful.
            
            # CRITICAL: The modified trainer code runs `rollout_func` ONLY on main process.
            # If you use colocate mode, you cannot simple use `trainer.llm.generate` on rank 0 
            # because `llm` is distributed.
            # FOR COLOCATE MODE: You must use the existing colocate logic in TRL which broadcasts.
            # OR, my trainer modification assumes you might want to run this everywhere if colocate.
            # FIX: My trainer modification specifically calls `rollout_func` on `accelerator.is_main_process`.
            # This implies `rollout_func` acts like a "controller" that might need to query a remote server.
            # IF using Colocate, you must modify the trainer to run rollout on ALL ranks or broadcast inputs/outputs inside rollout.
            # Assuming "Server" mode for this advanced rollout is safer and recommended.
            # If using Colocate, `trainer.llm.generate` inside main process will hang/fail if other ranks don't join.
            
            # Assuming Server mode or single-GPU debug:
            outputs = trainer.vllm_client.generate(vllm_prompts, **sampling_params)

        # Process outputs
        next_active_indices = []
        
        # If code execution is needed (check your logic for batching execution)
        # Here we do simplistic per-sample check
        code_snippets_to_exec = []
        exec_indices = []
        
        for idx, output_ids in zip(active_indices, outputs):
            # output_ids is a list of ints (single generation)
            if isinstance(output_ids, list) and isinstance(output_ids[0], list): 
                output_ids = output_ids[0] # Handle nested list if n=1 returns [[ids]]

            output_text = tokenizer.decode(output_ids)
            
            # Update history
            current_prompts_msgs[idx].append({"role": "assistant", "content": output_text})
            
            # Update completion IDs and Mask (1 for model)
            all_completion_ids[idx].extend(output_ids)
            all_completion_masks[idx].extend([1] * len(output_ids))
            
            # Check for code
            if output_text.strip().endswith(code_end_tag):
                # Extract code
                try:
                    code = output_text.split(code_start_tag)[-1].split(code_end_tag)[0]
                    code_snippets_to_exec.append(code)
                    exec_indices.append(idx)
                except:
                    # Malformed tags, stop
                    continue
            elif output_text.strip().endswith(tokenizer.eos_token):
                # Finished
                continue
            else:
                # Max length reached or other stop?
                continue

        # Execute Code
        if code_snippets_to_exec:
            # trainer.code_executor needs to be accessible
            # Assuming you attached it to trainer or it's a global
            execution_results = trainer.code_executor.batch_apply(code_snippets_to_exec)
            
            for idx, (res, report) in zip(exec_indices, execution_results):
                content = res if not report else f"Error: {report}"
                formatted_result = f"{res_start_tag}\n{content}\n{res_end_tag}"
                
                # Append to messages
                current_prompts_msgs[idx].append({"role": "user", "content": formatted_result})
                
                # Tokenize result
                result_ids = tokenizer.encode(formatted_result, add_special_tokens=False)
                
                # Append to completion IDs and Mask (0 for user/code result)
                all_completion_ids[idx].extend(result_ids)
                all_completion_masks[idx].extend([0] * len(result_ids))
                
                # Mark as active for next turn
                next_active_indices.append(idx)
        
        active_indices = next_active_indices

    # Finalize
    # prompt_ids are not strictly needed by TRL broadcasting if we return them, 
    # but for completeness we can return the tokenized initial prompts.
    
    # We return the dict. My modified trainer will pick up 'completion_mask'.
    return {
        "prompt_ids": [[] for _ in range(batch_size)], # Placeholder or actual IDs if needed
        "completion_ids": all_completion_ids,
        "completion_mask": all_completion_masks, 
        # Logprobs can be approximated or ignored (None) if not using PPO-style KL per token
        # GRPO often recalculates logprobs on the train forward pass.
        "logprobs": None 
    }