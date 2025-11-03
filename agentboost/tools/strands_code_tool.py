"""
Extract and execute Python code from the model's latest response with problem validation.

This tool automatically finds the problem statement, extracts sample test cases,
and validates the extracted code against those samples.
"""

import re
import sys
import time
import traceback
import subprocess
from io import StringIO
from typing import Dict, Any, List, Tuple, Optional

from strands import tool, ToolContext


class ProblemParseError(Exception):
    """Raised when problem statement cannot be parsed."""
    pass


def find_problem_statement(agent) -> str:
    """Find the original problem statement in conversation history."""
    for i, message in enumerate(agent.messages):
        if message.get("role") == "user":  
            content = message.get("content")
            
            if not content:  
                continue
            
            if isinstance(content, str):
                if content.strip():
                    return content
            
            elif isinstance(content, list):
                text_content = ""
                for item in content:
                    if isinstance(item, str):
                        text_content += item
                    elif isinstance(item, dict) and "text" in item:
                        text_content += item["text"]
                
                if text_content.strip():
                    return text_content
    
    raise ProblemParseError("No user message found in conversation history")

def parse_problem_statement(text: str) -> Dict[str, Any]:
    """Parse problem statement and extract timeout + examples."""
    
    timeout_match = re.search(r'Time Limit:\s*(\d+)\s*ms', text)
    timeout = float(timeout_match.group(1)) / 1000.0 if timeout_match else 5.0
    
    examples = []

    if "===BEGIN_PUBLIC_TESTS===" in text and "===END_PUBLIC_TESTS===" in text:
        test_section = text.split("===BEGIN_PUBLIC_TESTS===")[1].split("===END_PUBLIC_TESTS===")[0]
        
        test_pattern = r'TEST_(\d+)_INPUT:\n(.*?)\nTEST_\1_OUTPUT:\n(.*?)(?=\nTEST_|\s*$)'
        matches = re.findall(test_pattern, test_section, re.DOTALL)
        
        for test_num, input_text, output_text in matches:
            examples.append({
                "input": input_text.strip(),
                "output": output_text.strip()
            })
        
    return {
        "format": "standardized",
        "timeout": timeout,
        "examples": examples
    }
    
def extract_code_from_response(agent) -> str:
    """Extract the last Python code block from the most recent assistant message."""
    
    for message in reversed(agent.messages):
        if message["role"] == "assistant":
            response_text = ""
            for content in message["content"]:
                if "text" in content:
                    response_text += content["text"]
            
            if not response_text:
                continue
            
            all_matches = []
            
            # 1. Python-specific code blocks: ```python\n...\n```
            python_pattern = r'```python\s*\n(.*?)\n```'
            for match in re.finditer(python_pattern, response_text, re.DOTALL):
                all_matches.append((match.start(), match.group(1).strip()))
            
            # 2. Generic code blocks: ```\n...\n```
            generic_pattern = r'```\s*\n(.*?)\n```'
            for match in re.finditer(generic_pattern, response_text, re.DOTALL):
                # Skip if it's already captured as a python block
                code_content = match.group(1).strip()
                if not any(code_content == existing[1] for existing in all_matches):
                    all_matches.append((match.start(), code_content))
            
            # 3. HTML-style code blocks: <code>...</code>
            html_pattern = r'<code>(.*?)</code>'
            for match in re.finditer(html_pattern, response_text, re.DOTALL):
                code_content = match.group(1).strip()
                if not any(code_content == existing[1] for existing in all_matches):
                    all_matches.append((match.start(), code_content))
            
            # Sort by position and return the last (rightmost) match
            if all_matches:
                all_matches.sort(key=lambda x: x[0])  # Sort by position
                return all_matches[-1][1]  # Return the last match's code
    
    return None


def run_code_with_input(code: str, input_data: str, timeout: float) -> Tuple[str, bool, str]:
    """Run code with input data and return (output, success, error_msg)."""
    
    try:
        # Create a temporary script to run
        process = subprocess.Popen(
            [sys.executable, '-c', code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
        
        if process.returncode == 0:
            return stdout.strip(), True, ""
        else:
            return "", False, f"Runtime error: {stderr.strip()}"
            
    except subprocess.TimeoutExpired:
        process.kill()
        return "", False, f"Timeout: Code execution exceeded {timeout:.1f} seconds"
    except Exception as e:
        return "", False, f"Execution error: {str(e)}"


def validate_code_with_samples(code: str, examples: List[Dict], timeout: float) -> Dict[str, Any]:
    """Validate code against sample test cases."""
    
    if not examples:
        try:
            compile(code, '<string>', 'exec')
            return {
                "success": True,
                "message": "No sample test cases found - Cannot validate!\n⚠️ Try submitting if confident the solution is correct",
                "details": []
            }
        except SyntaxError as e:
            return {
                "success": False,
                "message": f"Syntax error in code: {e.msg} at line {e.lineno}",
                "details": [{"error": str(e), "line": e.lineno}]
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to compile code: {str(e)}",
                "details": [{"error": str(e)}]
            }
    
    start_time = time.time()
    results = []
    
    for i, example in enumerate(examples):
        elapsed = time.time() - start_time
        remaining_time = max(0.1, timeout - elapsed)
        
        if elapsed >= timeout:
            return {
                "success": False,
                "message": f"Total timeout exceeded: {timeout:.1f}s limit reached",
                "details": results
            }
        
        output, success, error = run_code_with_input(
            code, 
            example["input"], 
            remaining_time
        )
        
        expected = example["output"].strip()
        actual = output.strip()
        
        test_result = {
            "test_case": i + 1,
            "success": success and (actual == expected),
            "expected": expected,
            "actual": actual,
            "error": error
        }
        
        results.append(test_result)
        
        # If this test failed, return immediately
        if not test_result["success"]:
            total_time = time.time() - start_time
            if error:
                message = f"Test case {i+1} failed: {error}"
            else:
                message = f"Test case {i+1} failed:\nExpected: '{expected}'\nActual: '{actual}'"
                
                # Add diagnostics for common issues
                if actual.startswith(expected) and len(actual) > len(expected):
                    message += "\n⚠️ Partial match - your output might be correct with additional lines"
                elif sorted(actual.split()) == sorted(expected.split()):
                    message += "\n⚠️ Same values but different order - check if order matters"
            
            return {
                "success": False,
                "message": message,
                "details": results,
                "execution_time": total_time
            }
    
    # All tests passed
    total_time = time.time() - start_time
    return {
        "success": True,
        "message": f"✅ All {len(examples)} test cases passed in {total_time:.2f}s",
        "details": results,
        "execution_time": total_time
    }


@tool(context=True)
def strands_code_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Extract and execute Python code from the model's most recent response with problem validation.

    IMPORTANT: Do NOT pass code as a parameter. This tool automatically extracts Python code 
    from ```python blocks in your previous response. Call with no parameters: strands_code_tool()

    This tool automatically identifies the original problem statement in the conversation,
    extracts sample test cases, and validates the extracted code against those samples.
    Designed for competitive programming problems with automatic timeout detection.

    How It Works:
    ------------
    1. Finds the original problem statement in conversation history (first user message)
    2. Parses problem format (CodeForces or PrimeIntellect style) 
    3. Extracts timeout limits and sample input/output examples
    4. Extracts the latest Python code from ```python blocks in the assistant's response  
    5. Runs code against each sample with stdin/stdout simulation
    6. Validates output correctness and reports detailed results

    The tool automatically extracts test cases. If no test cases are found, it validates syntax only.

    Validation Process:
    -----------------
    - Runs extracted code against all sample test cases
    - Uses total timeout limit across all test cases (not per test case)
    - Compares actual output vs expected output (exact string match)
    - Reports first failure immediately or success summary

    Returns:
        Dict containing validation results in the format:
        {
            "status": "success|error",
            "content": [{"text": "Detailed validation results"}]
        }

        Success case: All sample test cases passed within time limit
        Error case: Test case failure, timeout, or parsing error

    Correct Usage:
    -------------
    Assistant: "Here's my solution:
    ```python
    [code]
    ```"
    Then call: strands_code_tool() if necessary or submit
    
    Tool automatically:
    1. Finds the original problem in conversation history
    2. Extracts sample inputs/outputs and timeout  
    3. Tests the assistant's code against samples
    4. Reports: "✅ All 3 test cases passed in 0.12s" or "❌ Test case 2 failed: expected '42', got '24'"

    Notes:
        - Call with NO parameters - code extraction is automatic
        - Requires problem statement in first user message of conversation
        - Timeout applies to total execution across all sample test cases
        - Output comparison is exact string matching (whitespace sensitive)
        - Perfect for validating competitive programming solutions
        - Supports both major problem statement formats
    """
    
    try:
        # Step 1: Find the original problem statement
        try:
            problem_text = find_problem_statement(tool_context.agent)
        except ProblemParseError as e:
            return {
                "status": "error",
                "content": [{"text": f"STATUS: TOOL_ERROR\n\nCould not find problem statement: {str(e)}"}]
            }
        
        # Step 2: Parse problem format and extract examples + timeout
        try:
            problem_data = parse_problem_statement(problem_text)
            timeout = problem_data["timeout"]
            examples = problem_data["examples"]
            format_type = problem_data["format"]
        except Exception as e:
            return {
                "status": "error", 
                "content": [{"text": f"Problem parsing failed: {str(e)}"}]
            }
        
        # Step 3: Extract code from the latest response
        code = extract_code_from_response(tool_context.agent)
        
        if not code:
            return {
                "status": "error",
                "content": [{"text": "No Python code blocks found in the recent response"}]
            }
        
        # Step 4: Show what we're about to test
        preview = (
            f"Found {format_type} format problem with {len(examples)} sample(s)\n"
            f"Timeout: {timeout:.1f}s total\n\n"
            f"Testing extracted code:\n```python\n{code}\n```\n\n"
        )
        
        # Step 5: Validate code against samples
        validation_result = validate_code_with_samples(code, examples, timeout)
        
        # Step 6: Format detailed results
        details = ""
        if validation_result.get("details"):
            details += "Test Results:\n"
            for result in validation_result["details"]:
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                details += f"  Test {result['test_case']}: {status}\n"
                if not result["success"]:
                    if result["error"]:
                        details += f"    Error: {result['error']}\n"
                    else:
                        details += f"    Expected: {result['expected']}\n"
                        details += f"    Actual:   {result['actual']}\n"
        
        exec_time = validation_result.get("execution_time", 0)
        summary = f"Execution time: {exec_time:.3f}s"

        if validation_result.get("details") is None or len(validation_result.get("details", [])) == 0:
            status_line = "STATUS: NO_VALIDATION\n\n"
        elif validation_result["success"]:
            status_line = "STATUS: VALIDATION_SUCCESS\n\n"
        else:
            status_line = "STATUS: TOOL_ERROR\n\n"
        
        result_text = status_line + preview + validation_result["message"] + "\n\n" + details + summary
        
        return {
            "status": "success" if validation_result["success"] else "error",
            "content": [{"text": result_text}]
        }
        
    except Exception as e:
        error_msg = f"Tool execution error: {str(e)}\n{traceback.format_exc()}"
        return {
            "status": "error", 
            "content": [{"text": error_msg}]
        }


if __name__ == "__main__":
    from strands import Agent

    agent = Agent(tools=[strands_code_tool])
    
    print("Strands Code Tool ready for competitive programming validation!")