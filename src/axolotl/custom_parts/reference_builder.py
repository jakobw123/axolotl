from typing import Any, Dict, List
from axolotl.custom_parts.parser import Parser
from axolotl.custom_parts.utils import PLAIN_TEXT_WRAPPER_ROLE, ROOT_ROLE, PrecompiledTemplate, RoleConfig, TreeNode, default_format_column


class ReferenceBuilder:
    # def __init__(
    #     self, 
    #     roles: Dict[str, RoleConfig],
    #     parser: Parser,
    #     data_role_map: Dict[str, str],
    #     data_is_pre_tagged: bool,
    #     output_template: str = None,
    # ):
    #     self.configure(
    #         roles=roles,
    #         parser=parser,
    #         data_role_map=data_role_map,
    #         data_is_pre_tagged=data_is_pre_tagged,
    #         output_template=output_template
    #     )
    
    def __init__(self):
        self.roles: Dict[str, RoleConfig] = None
        self.parser: Parser = None
        self.data_role_map: Dict[str, str] = None
        self.data_is_pre_tagged: bool = None
        self.output_template: str = None
        

    def configure(
        self, 
        roles: Dict[str, RoleConfig],
        parser: Parser,
        data_role_map: Dict[str, str],
        data_is_pre_tagged: bool,
        output_template: str = None,
    ):
        self.roles = roles
        self.role_mapping = data_role_map
        self.is_pre_tagged = data_is_pre_tagged
        self.parser = parser
        
        self.role_to_source_keys: Dict[str, List[str]] = {}
        if self.role_mapping:
            # role_mapping is { "dataset_column": "framework_role" }
            for source, target in self.role_mapping.items():
                if not self.role_to_source_keys.get(target):
                    self.role_to_source_keys[target] = []
                self.role_to_source_keys[target].append(source)
        
        self.output_template = None
        if output_template:
            self.output_template = PrecompiledTemplate(output_template)

    
    def build(self, normalized_example: Dict[str, Any]) -> TreeNode:
        res: TreeNode = None
        
        # Pre tagged string inside one single column
        if self.is_pre_tagged:
            res = self._build_from_tagged_string(normalized_example)
         
        # Use root role if present, it is assumed to contain all the data as segments
        # Use dataset's role mapping to map segments to roles
        elif ROOT_ROLE in self.role_to_source_keys:
            res = self._build_from_root(normalized_example)
        
        # Assemble roles from columns (use dataset's role mapping)
        else:
            res = self._build_from_mapping(normalized_example)

        if res:
            res.merge_consecutive_roles()
            return res

        return TreeNode(type=None)
    
    def _build_from_root(self, normalized_example: Dict[str, Any]) -> TreeNode:
        root = TreeNode(type=None)
        
        # If there is a ROOT_ROLE then constructors know it should contain everything and should only be one
        source_col = self.role_to_source_keys.get(ROOT_ROLE)[0]
        content = normalized_example.get(source_col)
        
        if content is not None:
            self._build_subtree(root, content)
                
        return root
    
    def _build_from_tagged_string(self, normalized_example: Dict[str, Any]) -> TreeNode:
        source_key = "solution"
        if self.role_mapping:
            source_key = next(iter(self.role_mapping.keys()))
        
        raw_text = normalized_example.get(source_key, "")
        
        if not isinstance(raw_text, str):
            raw_text = default_format_column(raw_text)
            
        return self.parser.parse(raw_text)
    
    def _build_from_mapping(self, row: Dict[str, Any]) -> TreeNode:
        root = TreeNode(type=None)
        
        if not self.role_mapping:
            return root
        
        for source_col, target_role_name in self.role_mapping.items():
            
            if target_role_name == ROOT_ROLE: 
                continue
            
            content = row.get(source_col)
            if content is None or content == "":
                continue
            
            if target_role_name in self.roles:
                node = TreeNode(type=target_role_name)
                self._build_subtree(node, content)
                root.add_part(node)
                
            elif target_role_name == PLAIN_TEXT_WRAPPER_ROLE:
                self._build_subtree(root, content)
            
            # For mapping this will just add columns available, 
            # while only the once in the mapping should be added
            # else:
            #     root.add_part(default_format_column(content))

        return root
    
    def _build_subtree(self, parent: TreeNode, content: Any):
        """
        Recursively populates a node with content.
        Handles Strings (Leafs) and Lists of Dicts (Branches).
        """
        # Base Case: Simple Text
        if isinstance(content, str):
            parent.add_part(content)
            
            return
        
        # Recursive Case 1: Single structural Dict
        if isinstance(content, dict):
            source_type = content.get("type") or content.get("role") or content.get("kind")
            inner_content = content.get("content") or content.get("value") or ""
            
            if not source_type:
                parent.add_part(default_format_column(content))
                return
            
            target_role_name = self.role_mapping.get(source_type)
            
            if target_role_name and target_role_name in self.roles:
                child = TreeNode(type=target_role_name)
                self._build_subtree(child, inner_content)
                parent.add_part(child)
                
            elif target_role_name == PLAIN_TEXT_WRAPPER_ROLE:
                self._build_subtree(parent, inner_content)
            
            else:
                parent.add_part(default_format_column(content))
                
            return

        # Recursive Case 2: Structural List
        if isinstance(content, list):
            
            for item in content:
                self._build_subtree(parent, item)
                
            return
           
        parent.add_part(default_format_column(content))