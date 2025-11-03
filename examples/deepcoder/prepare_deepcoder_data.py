import json

from datasets import concatenate_datasets, load_dataset
from datasets import disable_caching

from rllm.data.dataset import DatasetRegistry
from rllm.data.utils import fetch_live_code_bench_system_prompt

def prepare_deepcoder_data(train_size: int = None, test_size: int = None):
    train_dataset = concatenate_datasets([load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train")])
    test_dataset = concatenate_datasets([load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test")])

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        #question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)

        tests_raw = example["tests"]
        # Handle different test formats
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get("metadata", {})

        # Convert TACO format to standard format
        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
                normalized_tests.append({"input": input_val, "output": output_val, "testtype": "stdin_stdout"})
            tests = normalized_tests

        # Ensure tests is always a list
        if not isinstance(tests, list):
            tests = [tests] if tests else []

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}

        # Add standardized public tests to problem statement
        if len(tests) >= 2:  # Need at least 2 tests: 1 to show, 1 to evaluate
            # Simple rule: Show only the FIRST test, evaluate on the REST
            first_test = tests[0]
            remaining_tests = tests[1:]  # All others for evaluation
            
            input_str = str(first_test['input'])
            output_str = str(first_test['output'])
            
            # Only show if the first test isn't too long
            MAX_TEST_LENGTH = 1000
            if len(input_str) <= MAX_TEST_LENGTH and len(output_str) <= MAX_TEST_LENGTH:
                problem_with_tests = example["problem"] + "\n\n===BEGIN_PUBLIC_TESTS===\n"
                problem_with_tests += f"TEST_1_INPUT:\n{input_str}\nTEST_1_OUTPUT:\n{output_str}\n"
                problem_with_tests += "===END_PUBLIC_TESTS===\n"
                
                question = fetch_live_code_bench_system_prompt(problem_with_tests, starter_code if starter_code else None)
                return {
                    "question": question, 
                    "ground_truth": json.dumps(remaining_tests), 
                    "data_source": "livecodebench", 
                    "uid": f"deepcoder_{idx}", 
                    "index": idx, 
                    "starter_code": starter_code, 
                    "metadata": json.dumps(metadata)
                }
        # Fallback: Show no tests, evaluate on all
        question = fetch_live_code_bench_system_prompt(example["problem"], starter_code if starter_code else None)
        return {"question": question, "ground_truth": json.dumps(tests), "data_source": "livecodebench", "uid": f"deepcoder_{idx}", "index": idx, "starter_code": starter_code, "metadata": json.dumps(metadata)}

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    disable_caching()
    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    train_dataset = DatasetRegistry.register_dataset("agentcoder", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("agentcoder", test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_deepcoder_data()
    print(f"  - Train dataset: {len(train_dataset.get_data())} examples")
    print(f"  - Test dataset: {len(test_dataset.get_data())} examples")
    print(train_dataset.get_data()[0])
