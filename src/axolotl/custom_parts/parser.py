import re
from typing import Dict
from axolotl.custom_parts.utils import RoleConfig, TreeNode


class Parser:
    """
    Parses a raw model output string into a List[CanonicalBlock].
    
    This parser is order-preserving and handles interleaved/nested
    blocks by using a stack.
    """
    
    # def __init__(
    #     self, 
    #     output_roles: Dict[str, RoleConfig]
    # ):
    #     self.configure(output_roles=output_roles)
    
    def __init__(self):
        self.output_roles: Dict[str, RoleConfig] = None
        
    def configure(self, output_roles: Dict[str, RoleConfig]):
        self.output_roles = output_roles
        self.init_role_tag_maps()
        
    def init_role_tag_maps(self):
        tags = set()
        self.role_by_start_tag: Dict[str, str] = {}
        self.role_by_end_tag: Dict[str, str] = {}
        
        for role_name, role_config in self.output_roles.items():
            if not role_config.use_tags:
                continue
                
            if role_config.start_tag:
                tags.add(role_config.start_tag)
                self.role_by_start_tag[role_config.start_tag] = role_name
                
            if role_config.end_tag:
                tags.add(role_config.end_tag)
                self.role_by_end_tag[role_config.end_tag] = role_name
        
        sorted_tags = sorted(list(tags), key=len, reverse=True)
        
        if not sorted_tags:
            self.regex =  re.compile(r"($^)")
            
            return
        
        regex_str = '|'.join(map(re.escape, sorted_tags))
        
        self.regex = re.compile(f"({regex_str})")
        # self.regex = re.compile(f"({'|'.join(tags)})")

    def parse(self, model_output: str, merge_consecutive_equals: bool = False) -> TreeNode:
        root = TreeNode(type=None)
        current_node = root
        cursor = 0
        
        for match in self.regex.finditer(model_output):
            start_idx, end_idx = match.span()
            tag_str = match.group()
            
            if start_idx > cursor: 
                pre_content = model_output[cursor:start_idx]
                current_node.add_part(pre_content)
            
            start_role = self.role_by_start_tag.get(tag_str)
            end_role = self.role_by_end_tag.get(tag_str)
            
            # New semantic node
            if start_role:
                new_node = TreeNode(type=start_role)
                
                current_config = self.output_roles.get(current_node.type)
                current_node_is_atomic = not (current_config.allowed_children or []) if current_config else False
                
                while current_node_is_atomic and current_node.type is not None:
                    current_node = current_node.parent
                    current_config = self.output_roles.get(current_node.type)
                    current_node_is_atomic = not (current_config.allowed_children or [])  if current_config else False
                
                current_node.add_part(new_node) 
                current_node = new_node         
            
            # If closing tag of current role type is missing
            # this logic will close the current node and move on
            elif end_role:
                temp = current_node
                while temp.parent and temp.type != end_role:
                    temp = temp.parent
                
                if temp.type == end_role and temp.parent:
                    current_node = temp.parent

            cursor = end_idx

        if model_output[cursor:]:
            current_node.add_part(model_output[cursor:])
        
        if merge_consecutive_equals:
            root.merge_consecutive_roles()

        return root