import json
import multiprocessing
import io
import subprocess
import sys
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import signal
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from pebble import ProcessPool
from tqdm import tqdm
from functools import partial
from contextlib import redirect_stdout



_SANDBOX_PROC = None
_CFG_SANDBOX_PATH = None
_CFG_SERVER_PATH = None
_CFG_RUNTIME = None

def worker_init(sandbox_path: str, server_path: str, runtime: str):
    """Called once when worker starts. Stores config for restarts."""
    global _CFG_SANDBOX_PATH, _CFG_SERVER_PATH, _CFG_RUNTIME
    _CFG_SANDBOX_PATH = sandbox_path
    _CFG_SERVER_PATH = server_path
    _CFG_RUNTIME = runtime
    
    _start_sandbox_subprocess()

def _start_sandbox_subprocess():
    """
    Called once when a Pebble Worker starts.
    Spawns the persistent sandbox subprocess.
    """
    
    global _SANDBOX_PROC, _CFG_SANDBOX_PATH, _CFG_SERVER_PATH, _CFG_RUNTIME
    
    if _SANDBOX_PROC:
        try:
            _SANDBOX_PROC.kill()
            
        except: 
            pass
    
    try:
        spawn_ctx = multiprocessing.get_context("spawn")
        _SANDBOX_PROC = subprocess.Popen(
            [_CFG_SANDBOX_PATH, "-u", _CFG_SERVER_PATH, _CFG_RUNTIME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # stderr=sys.stderr,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    except Exception as e:
        pass
    
    
    # _SANDBOX_PROC = subprocess.Popen(
    #     [sandbox_python_path, "-u", server_script_path, runtime_name],
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True,
    #     bufsize=1
    # )

def worker_execute_task(task_payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Called for every task. Uses the existing persistent subprocess.
    """
    global _SANDBOX_PROC
    
    if _SANDBOX_PROC is None or _SANDBOX_PROC.poll() is not None:
        prev_err = _SANDBOX_PROC.stderr.read() if _SANDBOX_PROC else ""
        
        _start_sandbox_subprocess()
        
        if _SANDBOX_PROC is None or _SANDBOX_PROC.poll() is not None:
            return "", f"INFRASTRUCTURE FAILURE: Sandbox failed to start. {prev_err}"
        # dead_log = _SANDBOX_PROC.stderr.read() if _SANDBOX_PROC else "No process started"
        
        # print(f"\n[PIPELINE DEBUG] Worker is already dead. Error log:\n{dead_log}\n")
        
        # sys.exit(1)

    try:
        payload = json.dumps(task_payload) + "\n"
        _SANDBOX_PROC.stdin.write(payload)
        _SANDBOX_PROC.stdin.flush()
        
        while True:
            response_line = _SANDBOX_PROC.stdout.readline()
            
            if not response_line:
                crash_log = _SANDBOX_PROC.stderr.read()
                _SANDBOX_PROC = None 
                return "", f"INFRASTRUCTURE CRASH (Runtime): {crash_log}"
            
            cleaned = response_line.strip()
            if not cleaned: 
                continue
            
            try:
                data = json.loads(cleaned)
                return data.get("result", ""), data.get("error", "")
            
            except json.JSONDecodeError:
                # If we get non-JSON (like a warning print), ignore and wait for next line
                # Optional: print(f"Ignored garbage: {cleaned}", file=sys.stderr)
                continue
        
        # response_line = _SANDBOX_PROC.stdout.readline()
        
        # if not response_line:
        #     crash_log = _SANDBOX_PROC.stderr.read()
            
        #     print(f"\n[PIPELINE DEBUG] Worker crashed during execution!")
        #     print(f"--- STDERR START ---\n{crash_log}\n--- STDERR END ---")
            
        #     _SANDBOX_PROC = None 
            
        #     return "", f"INFRASTRUCTURE CRASH (Runtime): {crash_log}"
            
        # data = json.loads(response_line)
        
        # return data.get("result", ""), data.get("error", "")

    except Exception as e:
        _SANDBOX_PROC = None
        
        return "", f"INFRASTRUCTURE Error: {str(e)}"


class PythonExecutor:
    def __init__(
        self,
        sandbox_python_path: str = None,
        runtime_type: str = "generic",
        server_script_path: str = "executor_server.py",
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 10,
        hard_timeout_padding: int = 20,
        local_num_procs: int = 1
    ) -> None:
        self.sandbox_python_path = sandbox_python_path
        self.runtime_type = runtime_type
        self.server_script_path = server_script_path
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.soft_timeout = timeout_length
        self.hard_timeout = timeout_length + hard_timeout_padding
        self.local_num_procs = local_num_procs
        
        # DO NOT USE DEFAULT CONTEXT TO PREVENT FORK BOMB 
        spawn_ctx = multiprocessing.get_context("spawn")
        self.pool = ProcessPool(
            max_workers=self.local_num_procs, 
            context=spawn_ctx,
            max_tasks=100,
            initializer=worker_init,
            initargs=(self.sandbox_python_path, self.server_script_path, self.runtime_type)
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()
        
    def close(self):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()
        
    def process_generation_to_code(
        self, 
        gens: List[str]
    ) -> List[List[str]]:
        return [textwrap.dedent(g).strip().split("\n") for g in gens]

    @staticmethod
    def truncate(s: str, max_length: int = 800) -> str:
        half = max_length // 2
        
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
            
        return s
    
    # @staticmethod
    # def format_for_model(result: str, error: str) -> str:
    #     """
    #     Use this to decide what the model sees.
    #     - Hides system crashes.
    #     - Shows user code errors.
    #     """
    #     if "INFRASTRUCTURE" in error:
    #         # Log this for the admin!
    #         # print(f"Admin Log: {error}")
    #         return "System Error: Execution failed due to environment issues."
        
    #     if error:
    #         # Standard traceback -> The model can learn from this
    #         return f"Execution Error:\n{error}"
            
    #     return result

    def apply(
        self, 
        code: str
    ) -> Tuple[Any, Any]:
        return self.batch_apply([code])[0]

    def batch_apply(
        self, 
        batch_code: List[str]
    ) -> List[Tuple[Any, Any]]:
        all_code_snippets = self.process_generation_to_code(batch_code)
        
        tasks = [
            {
                "code": "\n".join(c), 
                "answer_symbol": self.answer_symbol,
                "answer_expr": self.answer_expr,
                "use_stdout": self.get_answer_from_stdout,
                "timeout": self.soft_timeout
            } 
            for c in all_code_snippets
        ]

        future = self.pool.map(
            worker_execute_task, 
            tasks, 
            timeout=self.hard_timeout
        )
        
        all_exec_results = []
        iterator = future.result()

        progress_bar = tqdm(total=len(all_code_snippets), desc="Execute") if len(all_code_snippets) > 100 else None

        while True:
            try:
                result = next(iterator)
                all_exec_results.append(result)
                
            except StopIteration:
                break
            
            except TimeoutError:
                all_exec_results.append(("", "Timeout Error (Hard)"))
                
            except Exception as error:
                print(f"Hard Timeout/Process Crash: {error}")
                all_exec_results.append(("", f"Execution Error (Hard): {str(error)}"))
                
            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            if not code or code == [""]:
                res, report = "empty code", "empty code"
                
            else:
                res, report = str(res).strip(), str(report).strip()
                res, report = self.truncate(res), self.truncate(report)
                
            batch_results.append((res, report))
            
        return batch_results


# def _test():
#     batch_code = [
#         """
#         for i in range(5):
#             if i % 2 == 0:
#                 print(f"{i} is even")
#         """,
#         """
#         print("Hello world3!")
#         print("Hello world4!")
#         print("Hello world5!")
#         """,
#         """
#         print("Hello world6!")
#         """,
#     ]

#     executor = PythonExecutor(get_answer_from_stdout=True)
#     predictions = executor.batch_apply(batch_code)
#     print(predictions)


# if __name__ == "__main__":
#     _test()
