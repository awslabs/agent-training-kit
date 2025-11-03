"""
Strands Worker Process
Runs in subprocess to handle Strands agent execution with full isolation.
Communicates via JSON over stdin/stdout.
"""

import json
import sys
import asyncio
import traceback
import logging
import importlib
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

try:
    from strands import Agent as StrandsAgentCore
    from strands.models.openai import OpenAIModel
    from strands.types.exceptions import MaxTokensReachedException, EventLoopException
    STRANDS_AVAILABLE = True
except ImportError as e:
    STRANDS_AVAILABLE = False
    logger.error(f"Strands not available: {e}")

from openai import APITimeoutError, Timeout
from httpx import ConnectTimeout
from jinja2.sandbox import SandboxedEnvironment
from agentboost.agents.hermes_tool_use_template import template_str

class StrandsWorker:
    """Handles Strands agent execution in subprocess - IDENTICAL to StrandsAgent behavior"""
    
    def __init__(self):
        self.strands_agent = None
        self._execution_error = None
        
    def reconstruct_tools(self, tool_configs):
        """Reconstruct actual tool modules from tool configs"""
        tools = []
        for config in tool_configs:
            try:
                module_name = config["module"]      # e.g. "strands_tools.calculator"
                tool_name = config["name"]          # e.g. "calculator"
                
                # Import the tool module directly
                tool_module = importlib.import_module(module_name)
                tools.append(tool_module)
                
                logger.debug(f"Reconstructed tool module: {tool_name} from {module_name}")
                
            except ImportError as e:
                logger.error(f"Failed to import tool module {module_name}: {e}")
                continue
        
        logger.debug(f"Reconstructed {len(tools)} tool modules")
        return tools
        
    def create_agent(self, config: Dict[str, Any]) -> None:
        """Create Strands agent from configuration - EXACT copy from StrandsAgent._init_strands_agent"""
        if not STRANDS_AVAILABLE:
            raise ImportError("Strands not available")
            
        base_url = config["base_url"]
        model_id = config["model_id"]
        tool_configs = config.get("tools", [])
        system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
        model_params = config.get("model_params", {})
        
        tools = self.reconstruct_tools(tool_configs)
        
        model = OpenAIModel(
            client_args={
                "api_key": "dummy", 
                "base_url": base_url,
                "timeout": Timeout(
                    connect=60.0,  #  Longer than default 5.0 for FSDP/vLLM
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
            tools=tools,  
            system_prompt=system_prompt,
            callback_handler=None,
        )

        async def empty_cleanup():
            pass

        self.cleanup_func = empty_cleanup
        
        if hasattr(self.strands_agent, 'model') and hasattr(self.strands_agent.model, 'client'):
            client = self.strands_agent.model.client
            
            if hasattr(client, 'close') and callable(getattr(client, 'close')):
                self.cleanup_func = client.close
            elif hasattr(client, 'aclose') and callable(getattr(client, 'aclose')):
                self.cleanup_func = client.aclose
    
    async def execute_conversation(self, user_message: str) -> Dict[str, Any]:
        """Execute conversation - EXACT copy from StrandsAgent._execute_full_conversation_and_prepare_replay"""
        if self.strands_agent is None:
            raise ValueError("Agent not created")
            
        try:
            strands_response = await self.strands_agent.invoke_async(user_message)
            final_response = str(strands_response)
            self._execution_error = None
            
        except (MaxTokensReachedException, EventLoopException, APITimeoutError, ConnectTimeout) as e:
            self._execution_error = e
            if isinstance(e, MaxTokensReachedException):
                final_response = "[CONTEXT_LIMIT_REACHED: Conversation too long. No space for the final answer.]"
            else:
                final_response = "[TIMEOUT_REACHED: Answering takes too long.]"
        except Exception as ex:
            self._execution_error = ex
            final_response = "[EXCEPTION encountered.]"
        
        try:
            full_conversation = self._extract_conversation_sync()
        except Exception as e:
            logger.warning(f"Failed to extract conversation: {e}")
            full_conversation = self._create_fallback_conversation(user_message, final_response)
        
        return {
            "success": self._execution_error is None,
            "messages": full_conversation,
            "response": final_response,
            "error": type(self._execution_error).__name__ if self._execution_error else None
        }
    
    def _extract_conversation_sync(self) -> list[dict[str, Any]]:
        """
        EXACT copy from StrandsAgent._extract_conversation_sync
        SYNC version of conversation extraction.
        Convert async calls to sync equivalents.
        """
        
        system_prompt = self._generate_system_prompt_sync()
        
        vllm_messages = self._convert_agent_messages_to_vllm(system_prompt)
        
        return vllm_messages
    
    def _generate_system_prompt_sync(self) -> str:
        """EXACT copy from StrandsAgent._generate_system_prompt_sync"""
        basic_system_prompt = self.strands_agent.system_prompt or "You are a helpful assistant."
        tool_specs = self._create_tool_specs_from_agent()
        
        if not tool_specs:
            return basic_system_prompt
        
        openai_tools = self._format_openai_tools(tool_specs)
        return self._generate_vllm_system_prompt(basic_system_prompt, openai_tools)
    
    def _create_tool_specs_from_agent(self) -> List[Dict[str, Any]]:
        """EXACT copy from StrandsAgent._create_tool_specs_from_agent"""
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
        """EXACT copy from StrandsAgent._format_openai_tools"""
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
        """EXACT copy from StrandsAgent._generate_vllm_system_prompt"""
        
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
            
            # Fallback if regex doesn't work
            return self._fallback_vllm_system_prompt(basic_system_prompt, tools)
            
        except Exception as e:
            logger.warning(f"Failed to load Hermes template ({e}), using fallback")
            return self._fallback_vllm_system_prompt(basic_system_prompt, tools)
    
    def _fallback_vllm_system_prompt(self, basic_system_prompt: str, tools: List[Dict[str, Any]]) -> str:
        """EXACT copy from StrandsAgent._fallback_vllm_system_prompt"""
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
        """EXACT copy from StrandsAgent._convert_agent_messages_to_vllm"""
        vllm_messages = []
        
        if system_prompt:
            vllm_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        agent_messages = self.strands_agent.messages
        
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
        """EXACT copy from StrandsAgent._convert_user_message"""
        for item in content:
            # Regular text message
            if 'text' in item:
                return {
                    "role": "user",
                    "content": item['text']
                }
            
            # Tool result message
            elif 'toolResult' in item:
                tool_result = item['toolResult']
                result_content = ""
                
                # Extract result text
                if 'content' in tool_result:
                    for result_item in tool_result['content']:
                        if 'text' in result_item:
                            result_content = result_item['text']
                            break
                
                # Format as vLLM expects
                if result_content:
                    return {
                        "role": "user",
                        "content": f"<tool_response>\n{result_content}\n</tool_response>"
                    }
        
        return None
    
    def _convert_assistant_message(self, content: List[Dict]) -> Optional[Dict[str, Any]]:
        """EXACT copy from StrandsAgent._convert_assistant_message"""
        text_parts = []
        tool_calls = []
        
        # Process all content items
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
    
    def _build_tool_call(self, tool_use: Dict) -> Optional[Dict[str, Any]]:
        """EXACT copy from StrandsAgent._build_tool_call"""
        name = tool_use.get('name')
        input_args = tool_use.get('input', {})
        
        if name:
            return {
                "name": name,
                "arguments": input_args
            }
        
        return None
    
    def _create_fallback_conversation(self, user_message: str, response: str) -> list[dict[str, Any]]:
        """
        EXACT copy from StrandsAgent._create_fallback_conversation
        Create a fallback conversation when normal extraction fails.
        Based on the original exception handling logic.
        """
        try:
            system_prompt = self._generate_system_prompt_sync()
        except Exception:
            system_prompt = "You are a helpful AI assistant."
        
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
        
    async def cleanup(self):
        """EXACT copy from StrandsAgent.cleanup"""
        try:
            await self.cleanup_func()
        except Exception as e:
            logger.debug(f"Error during StrandsWorker cleanup: {e}")


async def main():
    """Main worker loop - reads config from stdin, executes conversation, returns result to stdout"""
    worker = StrandsWorker()
    
    try:
        input_line = sys.stdin.readline().strip()
        if not input_line:
            raise ValueError("No input received")
            
        config = json.loads(input_line)
        
        required_fields = ["base_url", "model_id", "user_message"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        worker.create_agent(config)
        
        result = await worker.execute_conversation(config["user_message"])
        
        print(json.dumps(result))
        sys.stdout.flush()
        
    except Exception as e:
        error_result = {
            "success": False,
            "messages": [],
            "response": "[WORKER_ERROR encountered.]",
            "error": f"WorkerError: {str(e)}"
        }
        print(json.dumps(error_result))
        sys.stdout.flush()
        logger.error(f"Worker error: {e}")
        logger.error(traceback.format_exc())
        
    finally:
        try:
            await worker.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


if __name__ == "__main__":
    if not STRANDS_AVAILABLE:
        error_result = {
            "success": False,
            "messages": [],
            "response": "[STRANDS_NOT_AVAILABLE]",
            "error": "ImportError: Strands library not available"
        }
        print(json.dumps(error_result))
        sys.exit(1)
    
    asyncio.run(main())