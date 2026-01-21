import re
from typing import Any, Dict, List, Tuple, Union

from src.code_interpreter.python_executor import PythonExecutor


# def parse_ground_truth(example: Dict[str, Any], data_name: str) -> Tuple[Any, str]:
#     if "gt_cot" in example:
#         return example["gt_cot"], strip_string(example["gt"])

#     # parse ground truth
#     if data_name in ["math", "ocw"]:
#         gt_cot = example["solution"]
#         gt_ans = extract_answer(gt_cot)
#     elif data_name == "gsm8k":
#         gt_cot, gt_ans = example["answer"].split("####")
#     elif data_name == "gsm-hard":
#         gt_cot, gt_ans = example["code"], example["target"]
#     elif data_name == "svamp":
#         gt_cot, gt_ans = example["Equation"], example["Answer"]
#     elif data_name == "asdiv":
#         gt_cot = example["formula"]
#         gt_ans = re.sub(r"\(.*?\)", "", example["answer"])
#     elif data_name == "mawps":
#         gt_cot, gt_ans = None, example["target"]
#     elif data_name == "tabmwp":
#         gt_cot = example["solution"]
#         gt_ans = example["answer"]
#         if example["ans_type"] in ["integer_number", "decimal_number"]:
#             if "/" in gt_ans:
#                 gt_ans = int(gt_ans.split("/")[0]) / int(gt_ans.split("/")[1])
#             elif "," in gt_ans:
#                 gt_ans = float(gt_ans.replace(",", ""))
#             elif "%" in gt_ans:
#                 gt_ans = float(gt_ans.split("%")[0]) / 100
#             else:
#                 gt_ans = float(gt_ans)
#     elif data_name == "bbh":
#         gt_cot, gt_ans = None, example["target"]
#     else:
#         raise NotImplementedError(data_name)
#     # post process
#     gt_cot = str(gt_cot).strip()
#     gt_ans = strip_string(gt_ans)
#     return gt_cot, gt_ans


# def parse_question(example: Dict[str, Any], data_name: str) -> str:
#     question = ""
#     if data_name == "asdiv":
#         question = f"{example['body'].strip()} {example['question'].strip()}"
#     elif data_name == "svamp":
#         body = example["Body"].strip()
#         if not body.endswith("."):
#             body = body + "."
#         question = f'{body} {example["Question"].strip()}'
#     elif data_name == "tabmwp":
#         title_str = (
#             f'regarding "{example["table_title"]}" ' if example["table_title"] else ""
#         )
#         question = f"Read the following table {title_str}and answer a question:\n"
#         question += f'{example["table"]}\n{example["question"]}'
#         if example["choices"]:
#             question += (
#                 f' Please select from the following options: {example["choices"]}'
#             )
#     else:
#         for key in ["question", "problem", "Question", "input"]:
#             if key in example:
#                 question = example[key]
#                 break
#     assert question != ""
#     return question.strip()


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string: str) -> str:
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string: str) -> str:
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

# TODO: add tag variables, how?
def extract_answer(
    pred_str: str,
    res_start_tag: str = "```output",
    res_end_tag: str = "```"
) -> str:
    if "boxed" in pred_str:
        
        ans = pred_str.split("boxed")[-1]
        
        if len(ans) == 0:
            return ""
        
        elif ans[0] == "{":
            stack = 1
            a = ""
            
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                    
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                    
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
            
        pred = a
        
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
        
    elif extract_program_output(pred_str, res_start_tag=res_start_tag, res_end_tag=res_end_tag) != "":
        # fall back to program
        pred = extract_program_output(pred_str, res_start_tag=res_start_tag, res_end_tag=res_end_tag)
        
    else:  # use the last number
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str.replace(",", ""))
        
        if len(pred) >= 1:
            pred = pred[-1]
            
        else:
            pred = ""

    # multiple line
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
        
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
        
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
        
    pred = strip_string(pred)
    
    return pred


def extract_program(
    result: str, 
    start_tag: str = "```python", 
    end_tag: str = "```", 
    last_only: bool = True
) -> str:
    """
    extract the program after "```python", and before "```"
    """
    program = ""
    start = False
    for line in result.split("\n"):
        
        if line.startswith(start_tag):
            
            if last_only:
                program = ""  # only extract the last program
                
            else:
                program += "\n# ========\n"
            start = True
            
        elif line.startswith(end_tag):
            start = False
            
        elif start:
            program += line + "\n"
            
    return program


def extract_programs_and_outputs(
    text: str,
    start_tag: str = "```python", 
    end_tag: str = "```",
    res_start_tag: str = "```output", 
    res_end_tag: str = "```",
) -> List[Tuple[str, str]]:
    """
    Extract all Python code blocks and their corresponding output blocks from the text.
    Returns a list of tuples, each tuple contains (program, output).
    If a program has no output block, the output will be an empty string.
    Incomplete or empty blocks are skipped.
    """

    def dedent_code(code: str) -> str:
        if not code:
            return code

        lines = code.splitlines()
        min_indent = float("inf")
        
        for line in lines:
            stripped = line.lstrip()
            
            if stripped:
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)

        if min_indent == float("inf"):
            return code

        dedented_lines = []
        
        for line in lines:
            if line.strip():
                dedented_lines.append(line[min_indent:])
                
            else:
                dedented_lines.append(line)

        return "\n".join(dedented_lines)

    results = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        # Skip until we find a Python code block start
        while i < len(lines) and not lines[i].strip() == start_tag:
            i += 1

        if i >= len(lines):
            break  # No more Python code blocks

        # Start processing Python block
        i += 1  # Skip ```python line
        code_block = ""
        code_complete = False

        # Extract code until closing backticks
        while i < len(lines):
            if lines[i].strip() == end_tag:
                code_complete = True
                i += 1  # Skip closing backticks
                
                break
            
            code_block += lines[i] + "\n"
            i += 1

        # Skip incomplete or empty code blocks
        if not code_complete or not code_block.strip():
            continue

        code_block = dedent_code(code_block)

        # Now look for an output block
        j = i
        output_block = ""
        output_found = False

        # Skip until output block or another Python block
        while j < len(lines):
            
            if lines[j].strip() == res_start_tag:
                # Found potential output block
                j += 1  # Skip ```output marker
                output_tmp = ""
                output_complete = False

                # Extract output until closing backticks
                while j < len(lines):
                    if lines[j].strip() == res_end_tag:
                        output_complete = True
                        j += 1  # Skip closing backticks
                        
                        break
                    
                    output_tmp += lines[j] + "\n"
                    j += 1

                if output_complete:
                    output_block = output_tmp
                    output_found = True
                    i = j  # Update main pointer
                    
                    break
                # If incomplete, continue looking

            elif lines[j].strip() == start_tag:
                # Found another Python block first
                break

            j += 1

        # Add code-output pair to results
        results.append((code_block, output_block))

    return results


def extract_jupyter_like_program(
    result: str,
    start_tag: str = "```python", 
    end_tag: str = "```",
    res_start_tag: str = "```output", 
    res_end_tag: str = "```",
) -> str:
    """
    Extract and process programs from text, handling imports and errors appropriately.
    - For programs with errors: keep only import statements
    - For successful programs: remove print statements
    """
    final_program = ""

    programs_and_outputs = extract_programs_and_outputs(
        result,
        start_tag=start_tag,
        end_tag=end_tag,
        res_start_tag=res_start_tag,
        res_end_tag=res_end_tag
    )
    
    if len(programs_and_outputs) == 0:
        return final_program

    def extract_imports(program: str) -> str:
        """Extract only import statements from program"""
        import_lines = []
        
        for line in program.split("\n"):
            line = line.strip()
            
            if line.startswith("import ") or line.startswith("from "):
                import_lines.append(line)
                
        return "\n".join(import_lines) + "\n" if import_lines else ""

    def remove_prints(program: str) -> str:
        """Remove print statements from program"""
        cleaned_lines = []
        for line in program.split("\n"):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Skip lines that are just print statements
            if line.startswith("print"):
                continue
            
            if line.startswith("sp.pprint"):
                continue
            
            if line.startswith("sympy.pprint"):
                continue
            
            if line.startswith("pprint"):
                continue
            
            if line.startswith("rprint"):
                continue
            # # Handle print statements that might be part of other code
            # if 'print(' in line:
            #     # If print is not the main statement, keep the line but remove the print
            #     if not line.strip().startswith('print('):
            #         line = line.replace('print(', '# print(')
            cleaned_lines.append(line)
            
        return "\n".join(cleaned_lines) + "\n" if cleaned_lines else ""

    def is_error_output(output: str) -> bool:
        """Check if output contains error messages"""
        if output is None:
            return False
        
        error_keywords = [
            "error",
            "exception",
            "traceback",
        ]
        
        output = output.lower()
        
        return any(keyword in output for keyword in error_keywords)

    def is_import_error_output(output: str) -> bool:
        """Check if output contains error messages"""
        if output is None:
            return False
        
        error_keywords = [
            "ImportError",
        ]
        
        # output = output.lower()
        return any(keyword in output for keyword in error_keywords)

    # Process all programs except the last one
    prev_programs_and_outputs = programs_and_outputs[:-1]
    for program, output in prev_programs_and_outputs:
        if start_tag in program:
            continue
        
        if (program.strip() != "") and (not is_error_output(output)):
            removed_print_program = remove_prints(program)
            
            if "print" not in removed_print_program:
                final_program += removed_print_program

    # Process the last program and output
    last_program, last_output = programs_and_outputs[-1]
    if start_tag in last_program:
        last_program_start_pos = last_program.rfind(start_tag)
        last_program = last_program[last_program_start_pos + len(start_tag):]
        
    final_program += last_program

    return final_program


def extract_program_output(
    pred_str: str,
    res_start_tag: str = "```output", 
    res_end_tag: str = "```",
) -> str:
    """
    extract output between the last <res_start_tag>\n...\n<res_end_tag>
    """
    if res_start_tag not in pred_str:
        return ""
    
    if res_start_tag in pred_str:
        pred_str = pred_str.split(res_start_tag)[-1]
        
    if res_end_tag in pred_str:
        pred_str = pred_str.split(res_end_tag)[0]
        
    output = pred_str.strip()
    
    return output


def run_execute(
    executor: "PythonExecutor", 
    result: str, 
    prompt_type: str, 
    execute: bool = False,
    start_tag: str = "```python", 
    end_tag: str = "```",
    res_start_tag: str = "```output", 
    res_end_tag: str = "```",
) -> Tuple[str, str]:
    if not result or result == "error":
        return None, None
    report = None

    if "program_only" in prompt_type:
        prediction = extract_program_output(
            result,
            res_start_tag=res_start_tag,
            res_end_tag=res_end_tag
        )
        
    elif prompt_type in ["pot", "pal"] and execute:
        code = extract_program(
            result,
            start_tag=start_tag,
            end_tag=end_tag
        )
        prediction, report = executor.apply(code)
        
    else:
        prediction = extract_answer(result)

    prediction = strip_string(prediction)
    
    return prediction, report
