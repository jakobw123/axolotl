"""
Axolotl Specific Training Args
"""

from dataclasses import dataclass

from trl import GRPOConfig

from axolotl.core.training_args import AxolotlTrainingMixins
# from .......src.output_handler.utils import RoleConfig


@dataclass
class AxolotlGRPOConfig(AxolotlTrainingMixins, GRPOConfig):
    """Axolotl GRPO Config for GRPO training"""

    context_parallel_size: int | None = None
    
    # data_role_map_and_pretag: dict[str, tuple[dict[str, str], bool]] | None = None
    # output_roles: dict[str, RoleConfig] | None = None
    
    # use_code_executor: bool = False
    
    # max_turns: int = 5
    # sandbox_python_path: str | None = None
    # sandbox_script_path: str | None = None
    # local_num_procs: int = 1
