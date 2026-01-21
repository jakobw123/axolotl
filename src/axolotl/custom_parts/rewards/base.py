from abc import ABC, abstractmethod
from typing import Dict

from axolotl.custom_parts.parser import Parser
from axolotl.custom_parts.reference_builder import ReferenceBuilder


class BaseAdaptiveReward(ABC):
    
    @abstractmethod
    def compute_score(self, prompt, completion, parser, builder, task_type) -> float:
        pass
    
    def __init__(self, name: str = None):
        self.__name__ = name if name else self.__class__.__name__
        self.parser_map = {}
        self.ref_builder_map = {}
        self.vllm = None

    def set_context(self, parser_map: Dict[str, Parser], ref_builder_map: Dict[str, ReferenceBuilder]):
        """Called by Trainer to inject pre-configured tools."""
        self.parser_map = parser_map
        self.ref_builder_map = ref_builder_map

    def set_vllm_engine(self, engine):
        self.vllm = engine

    def get_tools(self, task_type: str):
        """Fast lookup for the correct tools for this sample."""
        p = self.parser_map.get(task_type, self.parser_map.get("default"))
        b = self.ref_builder_map.get(task_type, self.ref_builder_map.get("default"))
        
        return p, b

    def __call__(self, prompts: list, completions: list, **kwargs):
        rewards = []
        task_types = kwargs.get("task_type", ["default"] * len(completions))

        for prompt, completion, task_type in zip(prompts, completions, task_types):
            parser, builder = self.get_tools(task_type)
            
            if not parser or not builder:
                rewards.append(0.0)
                continue

            score = self.compute_score(prompt, completion, parser, builder, task_type)
            rewards.append(score)
            
        return rewards