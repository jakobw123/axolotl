import re
from typing import Any, Dict, Generator, List, Literal, Optional, Set, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field



PLAIN_TEXT_WRAPPER_ROLE = "__text_wrapper__"
ROOT_ROLE = "__root__"

class RolloutReturn(BaseModel):
    # Allow extra keys
    model_config = ConfigDict(extra='allow')

    completion_ids: list[list[int]] = Field(
        default_factory=list, 
        description="Token ids for completion part for multiple prompts."
    )
    prompt_ids: list[list[int]] = Field(
        default_factory=list,  
        description="Token ids for prompt part for multiple prompts."
    )
    logprobs: list[list[float]] = Field(
        default_factory=list, 
        description="Log probs for multiple prompts and completions."
    )


class RoleConfig(BaseModel):
    """Holds all configuration for a single output role."""
    name: Optional[str] = Field(
        default=None, 
        description="Internal identifier, auto-filled from config key."
    )
    
    # --- Schema & Structure Configuration ---
    schema_type: Literal["string", "object", "list_item"] = Field(
        default="string",
        description="Defines how this role is represented in the Pydantic schema. "
                    "'string': Simple field. 'object': Nested model. 'list_item': Part of the interleaved trace."
    )
    schema_fields: Optional[Dict[str, Any]] = Field(
        default=None,
        description="If schema_type is 'object' or 'list_item', defines the sub-fields. "
                    "Format: {'field_name': ('type_str', 'description')}"
    )
    allowed_children: List[str] = Field(
        default_factory=list,
        description="List of role names that can be logically nested inside this role."
    )
    
    # --- Formatting (Tags) ---
    start_tag: Optional[str] = Field(
        default=None, 
        description="Optional start tag for it's role."
    )
    end_tag: Optional[str] = Field(
        default=None, 
        description="Optional end tag for it's role."
    )
    content_format: Literal["text", "json"] = Field(
        default="text",
        description="Format to expect for parser (and prompter). At the moment either 'json' or 'text'. Defaults to 'text'."
    )
    
    # --- Instruction Generation ---
    use_tags: Optional[bool] = Field(
        default=True,
        description="If false, this role will not be tagged (even if tags are defined)."
    )
    use_instruction: Optional[bool] = Field(
        default=True,
        description="If false, this role's instruction will not be added."
    )
    behaviour_instruction: Optional[str] = Field(
        default=None, 
        description="Optional behaviour instructions."
    )
    format_instruction: Optional[str] = Field(
        default=None, 
        description="Optional format instructions."
    )
    
    use_tag_instruction: Optional[bool] = Field(
        default=True,
        description="If false, this role's tag instruction will not be shown."
    )


class TreeNode(BaseModel):
    """
    The single source of truth for a block in our sequence.
    Both ReferenceBuilder and Parser MUST output this structure.
    """
    type: Optional[str] = None
    parts: List[Union[str, "TreeNode"]] = Field(default_factory=list)
    parent: Optional["TreeNode"] = Field(default=None, exclude=True)
    
    @property
    def is_nested_role(self) -> bool:
        """
        Returns True if this node is a semantic Role (type is set) 
        AND it is a descendant of another semantic Role.
        """
        if self.type is None:
            return False
            
        current = self.parent
        while current:
            if current.type is not None:
                return True
            current = current.parent
        return False
    
    def add_part(self, part: Union[str, "TreeNode"]):
        """Adds a text segment or a child node to the sequence."""
        if isinstance(part, str):
            if not part: 
                return
            # Merge adjacent strings to reduce fragmentation
            if self.parts and isinstance(self.parts[-1], str):
                self.parts[-1] += part
            else:
                self.parts.append(part)
        else:
            part.parent = self
            self.parts.append(part)
            
    def get_text(
        self, 
        allowed_roles: Optional[Set[str]] = None, 
        add_text_from_root: bool = False,
        role_configs: Optional[Dict[str, RoleConfig]] = None,
        apply_tags: bool = True
    ) -> str:
        text_builder: List[str] = []
        
        is_root = self.type is None
        
        for part in self.parts:
            if isinstance(part, str):
                if (allowed_roles is None) or \
                    (self.type in allowed_roles) or \
                    (is_root and add_text_from_root):
                    text_builder.append(part)
                    
            elif isinstance(part, TreeNode):
                if (allowed_roles is None) or (part.type in allowed_roles):
                    text_builder.extend(part.get_text(allowed_roles=allowed_roles, role_configs=role_configs, apply_tags=apply_tags))
         
        inner_text = "".join(text_builder)
        
        if apply_tags and role_configs and self.type in role_configs:
            config = role_configs[self.type]
            use_tags = config.use_tags
            
            if use_tags:
                start = config.start_tag or ""
                end = config.end_tag or ""
                inner_text = f"{start}{inner_text}{end}"
                
        return inner_text

    def get_own_text(self, apply_tags: bool = False) -> str:
        """
        Returns only the text belonging directly to this node, skipping children.
        """
        if self.type is None:
            return self.get_text(allowed_roles=set(), apply_tags=False, add_text_from_root=True)

        return self.get_text(allowed_roles={self.type}, apply_tags=apply_tags)
    
    # def get_value(self, content_format: str = "text") -> Any:
    #     """
    #     Interprets the node's content based on its content_format.
    #     If format is 'json', attempts to parse the own_text.
    #     """
    #     raw_text = self.get_own_text()
        
    #     if content_format == 'json':
    #         try:
    #             return json.loads(raw_text.strip())
            
    #         except (json.JSONDecodeError, TypeError):
    #             return raw_text
                
    #     return raw_text

    def find_all(self, role: str) -> Generator["TreeNode", None, None]:
        """
        Recursive generator to find all descendants of a specific role.
        """
        if self.type == role:
            yield self
        
        for part in self.parts:
            if isinstance(part, TreeNode):
                yield from part.find_all(role)
                
    def _flatten(self) -> List["TreeNode"]:
        if not any(isinstance(p, TreeNode) for p in self.parts):
            return [TreeNode(type=self.type, parts=self.parts, parent=None)]
        
        flat_list = []
        
        current_segment = TreeNode(type=self.type)
        
        for part in self.parts:
            if isinstance(part, str):
                current_segment.add_part(part)
                
            elif isinstance(part, TreeNode):
                if current_segment.parts:
                    flat_list.append(current_segment)
                
                flat_list.extend(part._flatten())
                
                current_segment = TreeNode(type=self.type)
        
        if current_segment.parts:
            flat_list.append(current_segment)
            
        return flat_list
    
    def flat_interleaved(self) -> "TreeNode":
        flat_parts = self._flatten()
        
        flat_parts = [(p.parts[0] if not p.type else p) for p in flat_parts ]
        
        return TreeNode(type=None, parent=None, parts=flat_parts)
    
    def merge_consecutive_roles(self) -> "TreeNode":
        """
        Recursively merges adjacent sibling nodes of the same type.
        Example: [Think(A), Think(B), Code(C)] -> [Think(A+B), Code(C)]
        """
        if not self.parts:
            return

        new_parts = []
        last_node = None

        for part in self.parts:
            if isinstance(part, str):
                new_parts.append(part)
                last_node = None
                continue

            part.merge_consecutive_roles()

            if last_node and last_node.type == part.type and part.type is not None:
                for sub_part in part.parts:
                    last_node.add_part(sub_part)
            else:
                new_parts.append(part)
                last_node = part

        self.parts = new_parts
        
    # def nest_by_config(self, role_configs: Dict[str, RoleConfig]):
    #     """
    #     Restructures a flat list of siblings into a nested tree based on allowed_children.
    #     Heuristic: A node "swallows" subsequent siblings if they are allowed children.
        
    #     Example:
    #     Input: [Think, Code, Think]
    #     Config: Think allows Code.
    #     Output: [Think(Code), Think]  <-- The first Think swallowed the Code.
    #     """
    #     if not self.parts:
    #         return

    #     for part in self.parts:
    #         if isinstance(part, TreeNode):
    #             part.nest_by_config(role_configs)

    #     new_parts = []
    #     active_parent: Optional[TreeNode] = None

    #     for part in self.parts:
    #         if isinstance(part, str):
    #             if active_parent:
    #                 active_parent.add_part(part)
                    
    #             else:
    #                 new_parts.append(part)
                    
    #             continue

    #         # It's a TreeNode
    #         # Check if we have an active parent that can accept this child
    #         if active_parent:
    #             parent_config = role_configs.get(active_parent.type)
                
    #             if parent_config and part.type in parent_config.allowed_children:
    #                 # SWALLOW: Move 'part' inside 'active_parent'
    #                 active_parent.add_part(part)
                    
    #                 # NOTE: We do NOT update active_parent here. 
    #                 # The 'Think' stays open to swallow more 'Code'.
    #                 # Exception: If 'Code' is a container itself (like Refinement), 
    #                 # do we want 'Think' to stay open? Usually yes, Think wraps everything.
    #                 continue
    #             else:
    #                 active_parent = None

    #         # If we are here, 'part' was not swallowed. It becomes a new top-level sibling.
    #         new_parts.append(part)

    #         config = role_configs.get(part.type)
    #         if config and config.allowed_children:
    #             active_parent = part
    #         else:
    #             active_parent = None

    #     self.parts = new_parts
                
TreeNode.model_rebuild()


class PrecompiledTemplate:
    _format_pattern = re.compile(r"\{(\w+)\}")
    
    def __init__(self, template: str):
        self.template = template
        # Precompute positions of placeholders
        # self.placeholders = set(re.findall(self._format_pattern, template))
        self.slots = [(m.start(), m.end(), m.group(1)) for m in self._format_pattern.finditer(template)]
    
    def format(self, mapping: dict) -> str:
        if len(self.slots) == 0:
            return self.template
        
        # Build the result string manually
        last_idx = 0
        parts = []
        for start, end, key in self.slots:
            parts.append(self.template[last_idx:start])
            parts.append(str(mapping.get(key, "")))
            last_idx = end
        parts.append(self.template[last_idx:])
        
        return "".join(parts)

def default_format_column(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    elif isinstance(value, (list, tuple, set)):
        value = "\n".join([default_format_column(item) for item in value]).strip()
        
    elif isinstance(value, dict):
        value = "\n".join([str(k) + ": " + default_format_column(v) for k, v in value.items()]).strip()
        
    elif hasattr(value, 'tolist') and callable(value.tolist):
        value = value.tolist()
        return default_format_column(value)
            
    elif isinstance(value, (int, float, bool)):
        value = str(value).strip()
            
    else:
        value = ""

    return value