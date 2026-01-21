import copy
import pickle
import signal
import subprocess
import importlib
import sys
import json
import io
import re
import traceback
from contextlib import redirect_stdout
from typing import Any, Dict

PACKAGE_ALIASES = {
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "fitz": "pymupdf",
    "magic": "python-magic",
    "usb": "pyusb",
    "serial": "pyserial",
    "telegram": "python-telegram-bot"
}

class CodeTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise CodeTimeoutError("Soft Timeout")

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c, timeout=60)

    def exec_code(self, code_piece: str, timeout: int) -> None:
        if re.search(r"(\s|^)?input\(", code_piece) or re.search(
            r"(\s|^)?os.system\(", code_piece
        ):
            raise RuntimeError("Restricted function usage detected (input/os.system)")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                exec(code_piece, self._global_vars)
                return
            
            except ModuleNotFoundError as e:
                if attempt == max_retries - 1:
                    raise e
                
                missing_module = getattr(e, 'name', None)
                if not missing_module:
                    try:
                        missing_module = str(e).split("'")[-2]
                        
                    except IndexError:
                        raise e

                targets_to_try = []
                
                # root_module = missing_module.split(".")[0]
                
                # try:
                #     root_spec = importlib.util.find_spec(root_module)
                #     if root_spec is not None and root_module != missing_module:
                #         sys.stderr.write(f"[Sandbox] Error: Package '{root_module}' exists, but submodule '{missing_module}' not found. Stopping retry.\n")
                #         raise e
                    
                # except Exception:
                #     pass
                
                if missing_module in PACKAGE_ALIASES:
                    targets_to_try.append(PACKAGE_ALIASES[missing_module])
                
                targets_to_try.append(missing_module)
                
                if "." in missing_module:
                    targets_to_try.append(missing_module.split(".")[0])

                # Try to install candidates in order
                installed = False
                install_errors = []
                
                signal.alarm(0)
                
                try:
                
                    for target in targets_to_try:
                        try:
                            sys.stderr.write(f"[Sandbox] Attempting pip install {target}...\n")
                            
                            subprocess.check_call(
                                [sys.executable, "-m", "pip", "install", target],
                                stdout=subprocess.DEVNULL,
                                # stderr=sys.stderr
                                stderr=subprocess.DEVNULL
                            )
                            sys.stderr.write(f"[Sandbox] Installed {target}.\n")
                            installed = True
                            break
                        
                        except Exception as err:
                            install_errors.append(f"{target}: {err}")
                            
                finally:
                    signal.alarm(timeout)
                
                if not installed:
                    sys.stderr.write(f"[Sandbox] Auto-install failed. Errors: {'; '.join(install_errors)}\n")
                    raise e
                
                importlib.invalidate_caches()
                
            except Exception:
                raise

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
            
    def get_answer(self, symbol: str) -> Any:
        return self._global_vars.get(symbol)
    
    
RUNTIME_MAP = {
    "generic": GenericRuntime,
}

def main():
    runtime_name = "generic"
    if len(sys.argv) > 1:
        runtime_name = sys.argv[1]
        
    runtime_cls = RUNTIME_MAP.get(runtime_name)
    if not runtime_cls:
        sys.stderr.write(f"Unknown runtime: {runtime_name}\n")
        return

    try:
        runtime = runtime_cls()
        
    except Exception as e:
        sys.stderr.write(f"Runtime Init Failed: {e}\n")
        return
    
    signal.signal(signal.SIGALRM, timeout_handler)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            code_str = request.get("code", "")
            
            if not code_str.strip():
                sys.stdout.write(json.dumps({"result": "", "error": ""}) + "\n")
                sys.stdout.flush()
                continue
            
            code = code_str.split("\n")
            answer_symbol = request.get("answer_symbol")
            answer_expr = request.get("answer_expr")
            get_answer_from_stdout = request.get("use_stdout", False)
            soft_timeout = int(request.get("timeout", 10))
            
            result = ""
            report = ""

            try:
                signal.alarm(soft_timeout)
                
                if get_answer_from_stdout:
                    program_io = io.StringIO()
                    
                    with redirect_stdout(program_io):
                        runtime.exec_code("\n".join(code), timeout=soft_timeout)
                        
                    program_io.seek(0)
                    result = program_io.read()
                    
                elif answer_symbol:
                    runtime.exec_code("\n".join(code), timeout=soft_timeout)
                    result = runtime.get_answer(answer_symbol)
                    
                elif answer_expr:
                    runtime.exec_code("\n".join(code), timeout=soft_timeout)
                    result = runtime.eval_code(answer_expr)
                    
                else:
                    if len(code) > 1:
                        runtime.exec_code("\n".join(code[:-1]), timeout=soft_timeout)
                    
                    last_line = code[-1].strip()
                    if last_line:
                        try:
                            result = runtime.eval_code(last_line)
                            
                        except SyntaxError:
                            runtime.exec_code(code[-1], timeout=soft_timeout)
                            result = ""
                            
                    else:
                        result = ""
                        
                    # runtime.exec_code("\n".join(code[:-1]), timeout=soft_timeout)
                    # result = runtime.eval_code(code[-1])
                
                # serialization check
                try: 
                    pickle.dumps(result)
                    
                except: 
                    pass
                
                result = str(result)
                
            except CodeTimeoutError:
                result = ""
                report = "Timeout Error (Soft)"
                
            except Exception:
                result = ""
                report = traceback.format_exc()
                
            finally:
                signal.alarm(0)
            
            response = {"result": result, "error": report}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except KeyboardInterrupt:
            break
        
        except Exception as e:
            sys.stderr.write(f"Server Crash: {e}\n")
            break
        
        

if __name__ == "__main__":
    main()