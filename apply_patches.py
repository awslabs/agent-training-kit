import shutil
import os

def apply_modifications():
    # Check if rllm is installed and importable
    try:
        import rllm
    except ImportError:
        raise ImportError("rllm not installed or not importable. Please install rllm before applying patches.")
    
    # Check if rllm has __file__ attribute
    try:
        rllm_path = os.path.dirname(rllm.__file__)
    except AttributeError:
        raise ImportError("rllm not installed or not importable. Cannot determine rllm installation path.")
    
    # Check if source files exist
    source_files = [
        "modified_rllm_files/agent_execution_engine.py",
        "modified_rllm_files/agent_ppo_trainer.py",
        "modified_verl_files/vllm_async_server.py"
    ]
    
    for src in source_files:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source file not found: {src}. Please ensure you're running from the correct directory.")
    
    # Define file operations: (source, destination, backup_destination)
    file_operations = [
        (
            "modified_rllm_files/agent_execution_engine.py",
            f"{rllm_path}/engine/agent_execution_engine.py",
            f"{rllm_path}/engine/agent_execution_engine.py.backup"
        ),
        (
            "modified_rllm_files/agent_ppo_trainer.py",
            f"{rllm_path}/trainer/verl/agent_ppo_trainer.py",
            f"{rllm_path}/trainer/verl/agent_ppo_trainer.py.backup"
        ),
        (
            "modified_verl_files/vllm_async_server.py",
            f"{rllm_path}/../verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py",
            f"{rllm_path}/../verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py.backup"
        )
    ]
    
    # Check if destination directories exist
    for src, dst, backup in file_operations:
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            raise FileNotFoundError(f"Destination directory does not exist: {dst_dir}. Please check your rllm installation.")
    
    # Perform backup and copy operations with error handling
    for src, dst, backup in file_operations:
        try:
            # Backup original file
            if os.path.exists(dst):
                shutil.copy(dst, backup)
            
            # Copy modified version
            shutil.copy(src, dst)
        except PermissionError as e:
            raise PermissionError(f"Permission denied when accessing {dst}. Please ensure you have write permissions.") from e
        except OSError as e:
            if e.errno == 28:  # No space left on device
                raise OSError(f"Disk full: Cannot write to {dst}. Please free up disk space.") from e
            else:
                raise OSError(f"Error copying file {src} to {dst}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error while processing {src}: {e}") from e
    
    print(f"✓ Patches applied to {rllm_path}")
    print("  Original files backed up with .backup extension")

if __name__ == "__main__":
    apply_modifications()