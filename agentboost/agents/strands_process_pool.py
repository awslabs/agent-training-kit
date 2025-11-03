import asyncio
import json
import time
from typing import Dict, Any, Optional
import os

class StrandsProcessPool:
    """Simple admission control for process execution - no process reuse"""
    
    def __init__(self, pool_size: int = 32, default_timeout: int = 600):
        self.pool_size = pool_size
        self.default_timeout = default_timeout
        
        self.semaphore = None
        
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.timeout_executions = 0
    
    def _get_semaphore(self):
        """
        Create semaphore, recreate if event loop changed due to
        different Ray workers between batches
        """
        try:
            current_loop = id(asyncio.get_running_loop())
            
            if (self.semaphore is None or 
                getattr(self, '_semaphore_loop_id', None) != current_loop):
                self.semaphore = asyncio.Semaphore(self.pool_size)
                self._semaphore_loop_id = current_loop
            
            return self.semaphore
        except RuntimeError:
            self.semaphore = asyncio.Semaphore(self.pool_size)
            return self.semaphore
    
    async def execute_conversation(
        self, 
        config: Dict[str, Any], 
        user_message: str, 
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute conversation - create fresh process, use it, let it die"""
        
        timeout = timeout or self.default_timeout
        self.total_executions += 1
            
        try:
            result = await self._create_and_execute_fresh_process(config, user_message, timeout)
            
            if result.get("success", False):
                self.successful_executions += 1
            else:
                self.failed_executions += 1
                if result.get("error") == "ExecutionTimeout":
                    self.timeout_executions += 1
            
            return result
            
        except Exception as e:
            print(f"ERROR:EXECUTE - Unexpected error: {e}")
            self.failed_executions += 1
            return {
                "success": False,
                "messages": [],
                "response": "[POOL_ERROR]",
                "error": f"PoolError: {str(e)}"
            }
        finally:
            pass
    
    async def _create_and_execute_fresh_process(
        self, 
        config: Dict[str, Any], 
        user_message: str, 
        timeout: int
    ) -> Dict[str, Any]:
        """Create fresh process, execute, let it die"""

        
        worker_script = os.path.join(os.path.dirname(__file__), "strands_worker.py")
        if not os.path.exists(worker_script):
            raise FileNotFoundError(f"Worker script not found: {worker_script}")
        
        input_config = {**config, "user_message": user_message}
        input_json = json.dumps(input_config) + "\n"
        
        async with self._get_semaphore():
            try:
                # Create fresh process
                process = await asyncio.create_subprocess_exec(
                    "python3", worker_script,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                
                start_time = time.time()
                
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(input_json.encode()),
                    timeout=timeout
                )

                await process.wait()
                
                execution_time = time.time() - start_time
                print(f"DEBUG:CREATE_EXEC - Process finished in {execution_time:.2f}s")
                
                
                try:
                    result = json.loads(stdout_data.decode())
                    result["execution_time"] = execution_time
                    
                    if stderr_data and stderr_data.decode().strip():
                        print(f"DEBUG:CREATE_EXEC - Process stderr: {stderr_data.decode().strip()}")
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    print(f"ERROR:CREATE_EXEC - JSON parse error: {e}")
                    print(f"ERROR:CREATE_EXEC - Raw stdout: {stdout_data.decode()}")
                    return {
                        "success": False,
                        "messages": [],
                        "response": "[JSON_PARSE_ERROR]",
                        "error": "JsonParseError"
                    }
            
            except asyncio.TimeoutError:
                print(f"ERROR:CREATE_EXEC - Process timed out after {timeout}s")
                
                # Kill the timed-out process
                try:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                        print(f"DEBUG:CREATE_EXEC - Killed timed-out process")
                except Exception as kill_error:
                    print(f"ERROR:CREATE_EXEC - Error killing timed-out process: {kill_error}")
                
                return {
                    "success": False,
                    "messages": [],
                    "response": "[EXECUTION_TIMEOUT]",
                    "error": "ExecutionTimeout"
                }
            
            except Exception as e:
                print(f"ERROR:CREATE_EXEC - Error creating/executing process: {e}")
                
                try:
                    if 'process' in locals() and process.returncode is None:
                        process.kill()
                        await process.wait()
                except Exception:
                    pass
                
                return {
                    "success": False,
                    "messages": [],
                    "response": "[PROCESS_ERROR]",
                    "error": f"ProcessError: {str(e)}"
                }
    
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get simple stats"""
        available_slots = self.semaphore._value
        busy_slots = self.pool_size - available_slots
        
        return {
            "pool_size": self.pool_size,
            "available_slots": available_slots,
            "busy_slots": busy_slots,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "timeout_executions": self.timeout_executions,
            "success_rate": self.successful_executions / max(1, self.total_executions)
        }
    
    async def cleanup(self):
        """Nothing to cleanup - no persistent processes"""
        print("DEBUG:CLEANUP - No cleanup needed for admission-control-only pool")
        pass