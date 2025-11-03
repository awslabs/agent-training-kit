import logging
from typing import Any, List, Dict, Optional
import json
import uuid

from rllm.agents.agent import BaseAgent, Action, Step, Trajectory
from openai import Timeout

try:
    from strands import Agent as StrandsAgentCore
    from strands.models.openai import OpenAIModel
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False

from jinja2.sandbox import SandboxedEnvironment
from agentboost.agents.hermes_tool_use_template import template_str

logger = logging.getLogger(__name__)

class StrandsAgent(BaseAgent):
    """Agent that uses Strands internally but provides BaseAgent interface"""
    
    def __init__(self, 
                 base_url: str,
                 model_id: str,
                 tools: List = None,
                 system_prompt: str = "You are a helpful AI assistant.",
                 model_params: Dict = None,
                 pool=None,
                 **kwargs):
        
        if not STRANDS_AVAILABLE:
            raise ImportError("Strands not available")
        
        self.pool = pool
        self.subprocess_config = {
            "base_url": base_url,
            "model_id": model_id,
            "model_params": model_params or {},
            "system_prompt": system_prompt,
            "tools": [
                {
                    "name": tool.__name__.split('.')[-1],
                    "module": tool.__name__ 
                }
                for tool in (tools or [])
            ]
        }
        self._init_strands_agent(base_url, model_id, tools, system_prompt, model_params or {})
        self._ipc_result = None

        self._trajectory = Trajectory()
        self._chat_completions = []
        
        self._is_executed = False 
        self._initial_observation = None
        self._replay_conversations = [] 
        self._current_replay_step = 0  
        self._execution_error = None 
        
    def _init_strands_agent(self, base_url, model_id, tools, system_prompt, model_params):
        """Initialize internal Strands agent with static configuration - never modified after creation"""
        
        model = OpenAIModel(
            client_args={
                "api_key": "dummy", 
                "base_url": base_url,
                "timeout": Timeout(
                    connect=60.0,
                    read=600,  
                    write=600,     
                    pool=600     
                )
            },
            model_id=model_id,
            params=model_params or {}
        )
        
        self.model = model
        
        self.strands_agent = StrandsAgentCore(
            model=model,
            tools=tools or [],
            system_prompt=system_prompt,
            callback_handler=None,
        )

    
    async def cleanup(self):
        pass
    
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Return extracted conversation with proper initialization handling"""
    
        if not self._is_executed and self._initial_observation is not None:
            try:
                return self.get_initial_system_user_messages(self._initial_observation)
            except Exception as e:
                logger.warning(f"Failed to get initial messages: {e}")
                return []
        
        if self._is_executed and not self._replay_conversations:
            logger.warning("StrandsAgent executed but no replay conversations available")
            if self._initial_observation is not None:
                try:
                    return self.get_initial_system_user_messages(self._initial_observation)
                except Exception:
                    return []
            return []
        
        return self._chat_completions
    
    @property 
    def trajectory(self) -> Trajectory:
        """Return trajectory"""
        return self._trajectory
        
    def reset(self):
        """Reset for new episode"""
        self._trajectory = Trajectory()
        self._chat_completions = []
        
        self._is_executed = False
        self._initial_observation = None
        self._replay_conversations = []
        self._current_replay_step = 0
        self._execution_error = None
        
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Update after env step"""
        
        if self._initial_observation is None:
            self._initial_observation = observation
        
        current_step = self.get_current_state()
        if current_step:
            current_step.observation = observation
            current_step.reward = reward
            current_step.done = done
            current_step.info.update(info)
    
    def _extract_user_message_from_observation(self, observation):
        """Extract user message from observation, similar to ToolAgent logic"""
        if isinstance(observation, dict):
            return observation.get('question') or str(observation)
        return str(observation)
    
    def get_initial_system_user_messages(self, observation):
        """Get initial system + user messages for prompt length checking"""
        user_message = self._extract_user_message_from_observation(observation)
        system_prompt = self._generate_system_prompt_sync()
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    
    async def _execute_full_conversation_and_prepare_replay(self):
        """Execute the complete Strands conversation and prepare replay steps"""
        if self._is_executed:
            return
        
        if self._initial_observation is None:
            raise ValueError("No observation available for Strands execution")
        
        user_message = self._extract_user_message_from_observation(self._initial_observation)
        
        try:
            result = await self.pool.execute_conversation(self.subprocess_config, user_message)
            self._ipc_result = result
            
            if result.get('success', False):
                final_response = result.get('response', '')
                self._execution_error = None
            else:
                error_type = result.get('error', 'UnknownError')
                if 'Timeout' in error_type:
                    final_response = "[TIMEOUT_REACHED: Answering takes too long.]"
                elif 'Context' in error_type or 'Token' in error_type:
                    final_response = "[CONTEXT_LIMIT_REACHED: Conversation too long. No space for the final answer.]"
                else:
                    final_response = "[EXCEPTION encountered.]"
                self._execution_error = error_type
                
        except Exception as ex:
            self._execution_error = ex
            final_response = "[EXCEPTION encountered.]"

        try:
            full_conversation = self._extract_conversation_sync()
        except Exception as e:
            logger.warning(f"Failed to extract conversation: {e}")
            full_conversation = self._create_fallback_conversation(final_response)
        
        self._prepare_replay_conversations(full_conversation, final_response)
        self._is_executed = True
    
    def _prepare_replay_conversations(self, full_conversation, final_response):
        """Break full conversation into step-by-step replay conversations"""
        self._replay_conversations = []
        
        if len(full_conversation) <= 2:  
            self._replay_conversations = [full_conversation]
            return
        
        base_messages = []
        for msg in full_conversation:
            if msg['role'] in ['system', 'user']:
                base_messages.append(msg)
                if msg['role'] == 'user':  
                    break
        
        remaining_messages = full_conversation[len(base_messages):]
        current_conversation = base_messages.copy()
        
        i = 0
        while i < len(remaining_messages):
            msg = remaining_messages[i]
            current_conversation.append(msg)
            
            if msg['role'] == 'assistant':
                j = i + 1
                while j < len(remaining_messages) and remaining_messages[j]['role'] == 'user':
                    current_conversation.append(remaining_messages[j])
                    j += 1
                
                self._replay_conversations.append(current_conversation.copy())
                i = j
            else:
                i += 1

        if not self._replay_conversations:
            self._replay_conversations = [full_conversation]
        
        if current_conversation and len(current_conversation) > len(base_messages):
            if len(self._replay_conversations) == 0 or \
                    len(current_conversation) > len(self._replay_conversations[-1]):
                self._replay_conversations.append(current_conversation.copy())
    
    def is_replay_complete(self) -> bool:
        """Check if we've completed all replay steps"""
        if not self._is_executed:
            return False
        
        if not self._replay_conversations:
            logger.warning("StrandsAgent: No replay conversations prepared, replay should not be complete")
            return False
        
        return self._current_replay_step >= len(self._replay_conversations)
        
    async def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Handle model updates - execute on first call, then replay.
        This preserves your exact extraction logic but adds replay functionality.
        """
        
        if not self._is_executed:
            await self._execute_full_conversation_and_prepare_replay()
        
        if self._current_replay_step < len(self._replay_conversations):
            conversation = self._replay_conversations[self._current_replay_step]
            
            expected_response = ""
            for msg in reversed(conversation):
                if msg['role'] == 'assistant':
                    expected_response = msg['content']
                    break
            
            actual_response = expected_response
            self._current_replay_step += 1
        else:
            logger.warning("StrandsAgent replay step beyond available conversations")
            actual_response = response
            conversation = self._chat_completions
        

        self._chat_completions = conversation

        tool_calls_dict = self._parse_strands_tool_calls(actual_response, conversation)
        
        if not tool_calls_dict:
            tool_calls_dict = [{
                "id": str(uuid.uuid4()),
                "type": "function", 
                "function": {
                    "name": "finish",
                    "arguments": {"response": actual_response}
                }
            }]

        
        step = Step()
        step.chat_completions = conversation
        step.model_response = actual_response
        step.action = tool_calls_dict       
        
        self._trajectory.steps.append(step)
        
        return Action(action=tool_calls_dict)
    
    def _create_fallback_conversation(self, response: str) -> list[dict[str, Any]]:
        """
        Create a fallback conversation when normal extraction fails.
        Based on the original exception handling logic.
        """
        try:
            system_prompt = self._generate_system_prompt_sync()
        except Exception:
            system_prompt = "You are a helpful AI assistant."
        
        if self._initial_observation is not None:
            user_message = self._extract_user_message_from_observation(self._initial_observation)
        else:
            user_message = response if response and response.strip() else "Please solve this problem."
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        if response and response.strip():
            conversation.append({"role": "assistant", "content": response})
        
        for i, msg in enumerate(conversation):
            if not msg.get('content') or not msg['content'].strip():
                msg['content'] = f"[FALLBACK_{msg['role'].upper()}_MESSAGE]"
    
        return conversation
    
    def _extract_conversation_sync(self) -> list[dict[str, Any]]:
        """
        SYNC version of conversation extraction.
        Convert async calls to sync equivalents.
        """
        if self._ipc_result:
            return self._ipc_result.get('messages', []) 
        
        system_prompt = self._generate_system_prompt_sync()
        
        vllm_messages = self._convert_agent_messages_to_vllm(system_prompt)
        
        return vllm_messages
    
    def _generate_system_prompt_sync(self) -> str:
        """SYNC version of system prompt generation"""
        basic_system_prompt = self.strands_agent.system_prompt or "You are a helpful assistant."
        tool_specs = self._create_tool_specs_from_agent()
        
        if not tool_specs:
            return basic_system_prompt
        
        openai_tools = self._format_openai_tools(tool_specs)
        return self._generate_vllm_system_prompt(basic_system_prompt, openai_tools)
    
    def _create_tool_specs_from_agent(self) -> List[Dict[str, Any]]:
        """Create tool specs from agent's tool registry"""
        tool_specs = []
        all_tools_config = self.strands_agent.tool_registry.get_all_tools_config()
        
        for tool_name, tool_config in all_tools_config.items():
            tool_spec = {
                "name": tool_name,
                "description": tool_config.get("description", ""),
                "inputSchema": {
                    "json": tool_config.get("parameters", {})
                }
            }
            tool_specs.append(tool_spec)
        
        return tool_specs
    
    def _format_openai_tools(self, tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools in OpenAI format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool_spec["name"],
                    "description": tool_spec["description"],
                    "parameters": tool_spec["inputSchema"]["json"],
                },
            }
            for tool_spec in tool_specs
        ]
    
    def _generate_vllm_system_prompt(self, basic_system_prompt: str, tools: List[Dict[str, Any]]) -> str:
        """Generate the full vLLM system prompt using Hermes template"""
        
        if not tools:
            return basic_system_prompt
        
        try:
            tool_template = template_str.replace(
                "{{- '<|im_start|>system\n' }}",
                "{{- '<|im_start|>system\n' }}{% for message in messages %}{% if message.role == 'system' %}{{ message.content + '\n\n' }}{% endif %}{% endfor %}"
            )
            
            jinja_template = SandboxedEnvironment().from_string(tool_template)
            
            mock_messages = [
                {"role": "system", "content": basic_system_prompt}
            ]
            
            rendered = jinja_template.render(
                bos_token="",
                tools=tools,
                messages=mock_messages,
                add_generation_prompt=False
            )
            
            import re
            system_match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', rendered, re.DOTALL)
            
            if system_match:
                return system_match.group(1).strip()
            
            return self._fallback_vllm_system_prompt(basic_system_prompt, tools)
            
        except Exception as e:
            logger.warning(f"Failed to load Hermes template ({e}), using fallback")
            return self._fallback_vllm_system_prompt(basic_system_prompt, tools)
    
    def _fallback_vllm_system_prompt(self, basic_system_prompt: str, tools: List[Dict[str, Any]]) -> str:
        """Fallback system prompt generation"""
        vllm_system = f"{basic_system_prompt}\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        vllm_system += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        vllm_system += json.dumps(tools, separators=(',', ': '))
        vllm_system += "\n</tools>\n\n"
        vllm_system += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        vllm_system += "<tool_call>\n"
        vllm_system += '{"name": <function-name>, "arguments": <args-json-object>}\n'
        vllm_system += '</tool_call>'
        return vllm_system
    
    def _convert_agent_messages_to_vllm(self, system_prompt: str) -> List[Dict[str, Any]]:
        """Convert Strands internal messages to vLLM format"""
        vllm_messages = []
        
        if system_prompt:
            vllm_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        agent_messages = self._ipc_result.get('messages', []) if self._ipc_result else []
        
        for msg in agent_messages:
            role = msg.get('role')
            content = msg.get('content', [])
            
            if role == 'user':
                user_msg = self._convert_user_message(content)
                if user_msg:
                    vllm_messages.append(user_msg)
                    
            elif role == 'assistant':
                assistant_msg = self._convert_assistant_message(content)
                if assistant_msg:
                    vllm_messages.append(assistant_msg)
        
        return vllm_messages
    
    def _convert_user_message(self, content: List[Dict]) -> Optional[Dict[str, Any]]:
        """Convert user message from Strands format"""
        
        for item in content:
            if 'text' in item:
                return {
                    "role": "user",
                    "content": item['text']
                }
            
            elif 'toolResult' in item:
                tool_result = item['toolResult']
                result_content = ""
                
                if 'content' in tool_result:
                    for result_item in tool_result['content']:
                        if 'text' in result_item:
                            result_content = result_item['text']
                            break
                
                if result_content:
                    return {
                        "role": "user",
                        "content": f"<tool_response>\n{result_content}\n</tool_response>"
                    }
        
        return None
    
    def _convert_assistant_message(self, content: List[Dict]) -> Optional[Dict[str, Any]]:
        """Convert assistant message from Strands format"""
    
        text_parts = []
        tool_calls = []
        
        for item in content:
            if 'text' in item:
                text_parts.append(item['text'])
            elif 'toolUse' in item:
                tool_use = item['toolUse']
                tool_call = self._build_tool_call(tool_use)
                if tool_call:
                    tool_calls.append(tool_call)
        
        final_content = ""
        
        if text_parts:
            final_content = "".join(text_parts)
        
        for tool_call in tool_calls:
            if final_content and not final_content.endswith('\n'):
                final_content += "\n"
            final_content += f"\n<tool_call>\n{json.dumps(tool_call)}\n</tool_call>"
        
        if final_content.strip():
            return {
                "role": "assistant", 
                "content": final_content
            }
        
        return None
    
    def _parse_strands_tool_calls(self, response: str, conversation: list = None) -> list[dict]:
        """Parse Strands <tool_call> format into ToolAgent format with rewards"""
        import re, json, uuid
        
        tool_calls = []
        
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        
        matches = re.findall(pattern, response, re.DOTALL)

        response_pattern = r'<tool_response>\s*\n(.*?)\n</tool_response>'
        if conversation:
            conversation_text = "\n".join(msg.get('content', '') for msg in conversation)
            all_tool_responses = re.findall(response_pattern, conversation_text, re.DOTALL)
        else:
            all_tool_responses = []
        
        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)

                tool_reward = 0.0
                if all_tool_responses:
                    tool_reward = self._calculate_tool_reward_from_content(all_tool_responses[-1])
                
                tool_calls.append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": json.dumps(call_data["arguments"])
                    },
                    "reward": tool_reward  # NEW: Add reward
                })
            except Exception as e:
                continue
        
        return tool_calls
    
    def _calculate_tool_reward_from_content(self, response_content: str) -> float:
        """Extract reward from tool status"""
        if "STATUS: TOOL_ERROR" in response_content:
            return 0.0
        elif "STATUS: NO_VALIDATION" in response_content:
            return 0.1
        elif "STATUS: VALIDATION_SUCCESS" in response_content:
            return 1.0
        else:
            return 0.0
    
    def _build_tool_call(self, tool_use: Dict) -> Optional[Dict[str, Any]]:
        """Build tool call in vLLM format"""
        name = tool_use.get('name')
        input_args = tool_use.get('input', {})
        
        if name:
            return {
                "name": name,
                "arguments": input_args  # Convert "input" to "arguments"
            }
        
        return None