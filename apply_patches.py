import shutil
import rllm
import os

def apply_modifications():
    rllm_path = os.path.dirname(rllm.__file__)
    
    # Backup originals
    shutil.copy(f"{rllm_path}/engine/agent_execution_engine.py", 
                f"{rllm_path}/engine/agent_execution_engine.py.backup")
    
    # Copy modified versions
    shutil.copy("modified_rllm_files/agent_execution_engine.py",
                f"{rllm_path}/engine/agent_execution_engine.py")

    # Backup originals
    shutil.copy(f"{rllm_path}/trainer/verl/agent_ppo_trainer.py", 
                f"{rllm_path}/trainer/verl/agent_ppo_trainer.py.backup")
    
    # Copy modified versions
    shutil.copy("modified_rllm_files/agent_ppo_trainer.py",
                f"{rllm_path}/trainer/verl/agent_ppo_trainer.py")
    
    # Backup originals
    shutil.copy(f"{rllm_path}/../verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py", 
                f"{rllm_path}/../verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py.backup")
    
    # Copy modified versions
    shutil.copy("modified_verl_files/vllm_async_server.py",
                f"{rllm_path}/../verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py")
    
    print(f"✓ Patches applied to {rllm_path}")
    print("  Original files backed up with .backup extension")

if __name__ == "__main__":
    apply_modifications()