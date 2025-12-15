"""
ToolRL-style reward function for Tau-2 benchmark tasks.
Based on: "ToolRL: Reward is All Tool Learning Needs" (NeurIPS 2025)
"""

import json
from typing import Any, Dict, List

from rllm.rewards.reward_fn import RewardFunction, RewardOutput


class Tau2RewardFunction(RewardFunction):
    """
    ToolRL's reward design: Format + Correctness with three-level decomposition.
    
    Reward Structure:
        R_final = R_format + R_correct ∈ [-3, 4]
        
        - R_format ∈ {0, 1}: Structural compliance
        - R_correct ∈ [-3, 3]: Three-level tool call accuracy
            - Level 1: Tool name matching (Jaccard similarity)
            - Level 2: Parameter key matching (Jaccard similarity per tool)
            - Level 3: Parameter value matching (exact match count)
    
    Key Design Principles (from ToolRL paper):
        1. Correctness reward scale > Format reward scale (prevents reward hacking)
        2. Fine-grained decomposition (dense gradient signal)
        3. Order-agnostic matching (finds optimal alignment)
    """
    
    def __call__(self, task_info: Dict[str, Any], action: Any) -> RewardOutput:
        """
        Evaluate agent performance against expected actions.
        
        Args:
            task_info: Dictionary containing:
                - ground_truth: JSON string with expected actions
                - tool_calls_made: List of tool calls made by agent
            action: The final response/action from the agent (unused)
        
        Returns:
            RewardOutput with reward in [-3, 4] and metadata
        """
        # Parse ground truth
        eval_criteria_json = task_info.get("ground_truth", "{}")
        try:
            eval_criteria = json.loads(eval_criteria_json) if isinstance(eval_criteria_json, str) else eval_criteria_json
        except json.JSONDecodeError:
            return RewardOutput(reward=-3.0, metadata={"error": "Invalid JSON"})
        
        tool_calls_made = task_info.get("tool_calls_made", [])
        expected_actions = eval_criteria.get("actions", [])
        
        # Compute rewards
        format_reward, format_meta = self._compute_format_reward(tool_calls_made, expected_actions)
        correct_reward, correct_meta = self._compute_correctness_reward(tool_calls_made, expected_actions)
        
        final_reward = format_reward + correct_reward
        
        return RewardOutput(
            reward=final_reward,
            metadata={
                "format_reward": format_reward,
                "correctness_reward": correct_reward,
                "final_reward": final_reward,
                "num_expected": len(expected_actions),
                "num_predicted": len(tool_calls_made),
                **format_meta,
                **correct_meta,
            }
        )
    
    def _compute_format_reward(
        self, 
        predicted: List[Dict], 
        expected: List[Dict]
    ) -> tuple[float, Dict]:
        """
        Format reward: R_format ∈ {0, 1}
        
        Checks if output has all required fields in correct structure.
        """
        # No tools expected
        if len(expected) == 0:
            return (1.0, {"format_status": "correct_no_action"})
        
        # Tools expected but none called
        if len(predicted) == 0:
            return (0.0, {"format_status": "missing_tool_calls"})
        
        # Validate structure of each tool call
        for i, call in enumerate(predicted):
            if not isinstance(call, dict):
                return (0.0, {"format_status": f"invalid_type_at_{i}"})
            if "name" not in call:
                return (0.0, {"format_status": f"missing_name_at_{i}"})
            if "arguments" not in call:
                return (0.0, {"format_status": f"missing_arguments_at_{i}"})
            if not isinstance(call.get("arguments"), dict):
                return (0.0, {"format_status": f"invalid_arguments_at_{i}"})
        
        return (1.0, {"format_status": "valid"})
    
    def _compute_correctness_reward(
        self, 
        predicted: List[Dict], 
        expected: List[Dict]
    ) -> tuple[float, Dict]:
        """
        Correctness reward: R_correct ∈ [-3, 3]
        
        Three-level decomposition:
            1. Tool name matching (Jaccard)
            2. Parameter key matching (Jaccard per matched tool)
            3. Parameter value matching (exact match count)
        """
        P, G = predicted, expected
        
        # Edge cases
        if len(G) == 0 and len(P) == 0:
            return (3.0, {"match_status": "correct_empty"})
        if len(G) == 0 and len(P) > 0:
            return (-3.0, {"match_status": "unexpected_calls"})
        if len(G) > 0 and len(P) == 0:
            return (-3.0, {"match_status": "missing_calls"})
        
        # Compute scores
        r_name = self._compute_name_score(P, G)
        r_param, r_value, matches = self._compute_param_scores(P, G)
        
        # Maximum possible score
        S_max = 1.0 + len(G) + sum(len(g.get("arguments", {})) for g in G)
        
        # Actual score
        R_match = r_name + r_param + r_value
        
        # Normalize to [-3, 3]
        correctness_reward = 6.0 * (R_match / S_max) - 3.0
        
        return (correctness_reward, {
            "match_status": "computed",
            "r_name": round(r_name, 4),
            "r_param": round(r_param, 4),
            "r_value": round(r_value, 4),
            "R_match": round(R_match, 4),
            "S_max": S_max,
            "matched_tools": matches,
        })
    
    def _compute_name_score(self, P: List[Dict], G: List[Dict]) -> float:
        """
        Level 1: Tool name matching using Jaccard similarity.
        
        r_name = |N_G ∩ N_P| / |N_G ∪ N_P| ∈ [0, 1]
        """
        N_G = set(g.get("name", "") for g in G)
        N_P = set(p.get("name", "") for p in P)
        
        union = N_G | N_P
        if not union:
            return 1.0
        
        return len(N_G & N_P) / len(union)
    
    def _compute_param_scores(
        self, 
        P: List[Dict], 
        G: List[Dict]
    ) -> tuple[float, float, int]:
        """
        Level 2 & 3: Parameter key and value matching.
        
        For each ground truth tool, find the best matching predicted tool
        (greedy matching by tool name).
        
        Returns:
            (r_param, r_value, num_matched_tools)
        """
        r_param = 0.0
        r_value = 0.0
        matches = 0
        used_predictions = set()
        
        for g in G:
            g_name = g.get("name", "")
            g_args = g.get("arguments", {})
            
            best_param = 0.0
            best_value = 0.0
            best_idx = -1
            
            # Find best matching prediction with same name
            for idx, p in enumerate(P):
                if idx in used_predictions:
                    continue
                if p.get("name", "") != g_name:
                    continue
                
                p_args = p.get("arguments", {})
                
                # Level 2: Parameter key matching (Jaccard)
                g_keys = set(g_args.keys())
                p_keys = set(p_args.keys())
                
                if not (g_keys | p_keys):
                    param_score = 1.0
                else:
                    param_score = len(g_keys & p_keys) / len(g_keys | p_keys)
                
                # Level 3: Parameter value matching (exact)
                value_score = sum(
                    1 for k in g_keys
                    if k in p_args and self._values_match(g_args[k], p_args[k])
                )
                
                # Track best match for this ground truth tool
                if param_score + value_score > best_param + best_value:
                    best_param = param_score
                    best_value = value_score
                    best_idx = idx
            
            # Record best match
            if best_idx >= 0:
                used_predictions.add(best_idx)
                r_param += best_param
                r_value += best_value
                matches += 1
        
        return r_param, r_value, matches
    
    def _values_match(self, expected: Any, predicted: Any) -> bool:
        """
        Check if two parameter values match.
        Handles type coercion for common cases.
        """
        # Direct equality
        if expected == predicted:
            return True
        
        # String comparison (handles int/float to string)
        if str(expected) == str(predicted):
            return True
        
        # Handle list comparison (order-independent for simple values)
        if isinstance(expected, list) and isinstance(predicted, list):
            try:
                return set(str(x) for x in expected) == set(str(x) for x in predicted)
            except TypeError:
                # Unhashable types, fall back to sorted comparison
                return sorted(str(x) for x in expected) == sorted(str(x) for x in predicted)
        
        # Handle nested dict comparison
        if isinstance(expected, dict) and isinstance(predicted, dict):
            if set(expected.keys()) != set(predicted.keys()):
                return False
            return all(
                self._values_match(expected[k], predicted[k]) 
                for k in expected.keys()
            )
        
        return False