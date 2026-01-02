"""
Model Version Control System.

Provides Git-like version control for machine learning models,
tracking changes, enabling rollbacks, and managing model evolution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import pickle
import shutil
from pathlib import Path
import difflib
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ModelCommit:
    """Represents a model version commit."""
    commit_id: str
    model_id: str
    parent_commit_id: Optional[str]
    timestamp: datetime
    author: str
    message: str
    
    # Changes
    parameter_changes: Dict[str, Any]
    architecture_changes: Dict[str, Any]
    performance_changes: Dict[str, float]
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    branch: str = "main"
    
    # Validation
    is_validated: bool = False
    validation_results: Optional[Dict[str, Any]] = None


@dataclass
class ModelBranch:
    """Represents a branch in model development."""
    branch_name: str
    created_at: datetime
    created_by: str
    base_commit_id: str
    current_commit_id: str
    description: str
    is_active: bool = True
    merge_request_id: Optional[str] = None


@dataclass
class ModelDiff:
    """Represents differences between model versions."""
    from_commit: str
    to_commit: str
    
    # Parameter differences
    param_added: Dict[str, Any]
    param_removed: Dict[str, Any]
    param_modified: Dict[str, Tuple[Any, Any]]  # old_value, new_value
    
    # Architecture differences
    arch_added: Dict[str, Any]
    arch_removed: Dict[str, Any]
    arch_modified: Dict[str, Tuple[Any, Any]]
    
    # Performance differences
    perf_changes: Dict[str, Tuple[float, float, float]]  # old, new, delta
    
    # Summary
    summary: str


class ModelVersionControl:
    """
    Version control system for machine learning models.
    
    Features:
    - Commit-based versioning
    - Branching and merging
    - Diff generation
    - Rollback capabilities
    - Tag management
    - Performance tracking across versions
    """
    
    def __init__(self, vcs_path: str = "model_vcs"):
        """
        Initialize version control system.
        
        Args:
            vcs_path: Base path for version control storage
        """
        self.vcs_path = Path(vcs_path)
        self.vcs_path.mkdir(parents=True, exist_ok=True)
        
        # VCS components
        self.commits_path = self.vcs_path / "commits"
        self.objects_path = self.vcs_path / "objects"
        self.branches_path = self.vcs_path / "branches"
        self.tags_path = self.vcs_path / "tags"
        
        # Create directories
        self.commits_path.mkdir(exist_ok=True)
        self.objects_path.mkdir(exist_ok=True)
        self.branches_path.mkdir(exist_ok=True)
        self.tags_path.mkdir(exist_ok=True)
        
        # Load VCS state
        self.commits = self._load_commits()
        self.branches = self._load_branches()
        self.tags = self._load_tags()
        
        # Current state
        self.current_branch = "main"
        self.head = self._get_head()
        
        logger.info(f"Model VCS initialized at {self.vcs_path}")
    
    def _load_commits(self) -> Dict[str, ModelCommit]:
        """Load all commits from disk."""
        commits = {}
        
        for commit_file in self.commits_path.glob("*.json"):
            try:
                with open(commit_file, 'r') as f:
                    commit_data = json.load(f)
                
                # Convert timestamp
                commit_data['timestamp'] = datetime.fromisoformat(commit_data['timestamp'])
                
                commit = ModelCommit(**commit_data)
                commits[commit.commit_id] = commit
                
            except Exception as e:
                logger.error(f"Failed to load commit {commit_file}: {e}")
        
        return commits
    
    def _load_branches(self) -> Dict[str, ModelBranch]:
        """Load all branches from disk."""
        branches = {}
        branches_file = self.branches_path / "branches.json"
        
        if branches_file.exists():
            with open(branches_file, 'r') as f:
                branches_data = json.load(f)
            
            for branch_name, branch_data in branches_data.items():
                # Convert timestamps
                branch_data['created_at'] = datetime.fromisoformat(branch_data['created_at'])
                branches[branch_name] = ModelBranch(**branch_data)
        else:
            # Create default main branch
            main_branch = ModelBranch(
                branch_name="main",
                created_at=datetime.now(),
                created_by="system",
                base_commit_id="",
                current_commit_id="",
                description="Main development branch"
            )
            branches["main"] = main_branch
            self._save_branches(branches)
        
        return branches
    
    def _load_tags(self) -> Dict[str, str]:
        """Load tags (tag_name -> commit_id mapping)."""
        tags = {}
        tags_file = self.tags_path / "tags.json"
        
        if tags_file.exists():
            with open(tags_file, 'r') as f:
                tags = json.load(f)
        
        return tags
    
    def _save_branches(self, branches: Dict[str, ModelBranch]):
        """Save branches to disk."""
        branches_data = {}
        
        for branch_name, branch in branches.items():
            branch_dict = branch.__dict__.copy()
            branch_dict['created_at'] = branch.created_at.isoformat()
            branches_data[branch_name] = branch_dict
        
        branches_file = self.branches_path / "branches.json"
        with open(branches_file, 'w') as f:
            json.dump(branches_data, f, indent=2)
    
    def _save_tags(self):
        """Save tags to disk."""
        tags_file = self.tags_path / "tags.json"
        with open(tags_file, 'w') as f:
            json.dump(self.tags, f, indent=2)
    
    def _get_head(self) -> Optional[str]:
        """Get current HEAD commit."""
        if self.current_branch in self.branches:
            return self.branches[self.current_branch].current_commit_id
        return None
    
    def generate_commit_id(self) -> str:
        """Generate unique commit ID."""
        timestamp = datetime.now().isoformat()
        random_data = np.random.bytes(16)
        hash_input = f"{timestamp}_{random_data.hex()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def commit(
        self,
        model: Any,
        model_id: str,
        author: str,
        message: str,
        parameters: Dict[str, Any],
        architecture: Dict[str, Any],
        performance: Dict[str, float],
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new model version commit.
        
        Args:
            model: Model instance
            model_id: Model ID
            author: Commit author
            message: Commit message
            parameters: Model parameters/hyperparameters
            architecture: Model architecture details
            performance: Performance metrics
            tags: Optional tags
            
        Returns:
            Commit ID
        """
        # Generate commit ID
        commit_id = self.generate_commit_id()
        
        # Get parent commit
        parent_commit_id = self.head
        
        # Calculate changes if parent exists
        parameter_changes = {}
        architecture_changes = {}
        performance_changes = {}
        
        if parent_commit_id and parent_commit_id in self.commits:
            parent_commit = self.commits[parent_commit_id]
            
            # Load parent model data
            parent_data = self._load_commit_data(parent_commit_id)
            if parent_data:
                parameter_changes = self._calculate_param_changes(
                    parent_data.get('parameters', {}), parameters
                )
                architecture_changes = self._calculate_param_changes(
                    parent_data.get('architecture', {}), architecture
                )
                performance_changes = self._calculate_performance_changes(
                    parent_data.get('performance', {}), performance
                )
        
        # Create commit
        commit = ModelCommit(
            commit_id=commit_id,
            model_id=model_id,
            parent_commit_id=parent_commit_id,
            timestamp=datetime.now(),
            author=author,
            message=message,
            parameter_changes=parameter_changes,
            architecture_changes=architecture_changes,
            performance_changes=performance_changes,
            tags=tags or [],
            branch=self.current_branch
        )
        
        # Save model object
        model_path = self.objects_path / f"{commit_id}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save commit data
        commit_data = {
            'parameters': parameters,
            'architecture': architecture,
            'performance': performance
        }
        data_path = self.objects_path / f"{commit_id}_data.json"
        with open(data_path, 'w') as f:
            json.dump(commit_data, f, indent=2)
        
        # Save commit
        self._save_commit(commit)
        
        # Update branch
        if self.current_branch in self.branches:
            self.branches[self.current_branch].current_commit_id = commit_id
            self._save_branches(self.branches)
        
        # Update HEAD
        self.head = commit_id
        
        # Add to commits
        self.commits[commit_id] = commit
        
        logger.info(f"Created commit {commit_id}: {message}")
        
        return commit_id
    
    def _save_commit(self, commit: ModelCommit):
        """Save commit to disk."""
        commit_dict = commit.__dict__.copy()
        commit_dict['timestamp'] = commit.timestamp.isoformat()
        
        commit_file = self.commits_path / f"{commit.commit_id}.json"
        with open(commit_file, 'w') as f:
            json.dump(commit_dict, f, indent=2)
    
    def _load_commit_data(self, commit_id: str) -> Optional[Dict[str, Any]]:
        """Load commit data from disk."""
        data_path = self.objects_path / f"{commit_id}_data.json"
        
        if data_path.exists():
            with open(data_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def _calculate_param_changes(self, old_params: Dict[str, Any], 
                               new_params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate parameter changes between versions."""
        changes = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        # Find added parameters
        for key in new_params:
            if key not in old_params:
                changes['added'][key] = new_params[key]
        
        # Find removed parameters
        for key in old_params:
            if key not in new_params:
                changes['removed'][key] = old_params[key]
        
        # Find modified parameters
        for key in old_params:
            if key in new_params and old_params[key] != new_params[key]:
                changes['modified'][key] = {
                    'old': old_params[key],
                    'new': new_params[key]
                }
        
        return changes
    
    def _calculate_performance_changes(self, old_perf: Dict[str, float],
                                     new_perf: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance changes between versions."""
        changes = {}
        
        for metric in new_perf:
            if metric in old_perf:
                delta = new_perf[metric] - old_perf[metric]
                relative_change = delta / (abs(old_perf[metric]) + 1e-8)
                changes[metric] = {
                    'old': old_perf[metric],
                    'new': new_perf[metric],
                    'delta': delta,
                    'relative_change': relative_change
                }
            else:
                changes[metric] = {
                    'old': None,
                    'new': new_perf[metric],
                    'delta': new_perf[metric],
                    'relative_change': None
                }
        
        return changes
    
    def checkout(self, target: str) -> bool:
        """
        Checkout a commit or branch.
        
        Args:
            target: Commit ID, branch name, or tag
            
        Returns:
            Success status
        """
        # Check if target is a branch
        if target in self.branches:
            self.current_branch = target
            self.head = self.branches[target].current_commit_id
            logger.info(f"Switched to branch {target}")
            return True
        
        # Check if target is a tag
        if target in self.tags:
            target = self.tags[target]
        
        # Check if target is a commit
        if target in self.commits:
            self.head = target
            logger.info(f"HEAD is now at {target}")
            return True
        
        logger.error(f"Target {target} not found")
        return False
    
    def create_branch(self, branch_name: str, description: str, 
                     author: str, base_commit: Optional[str] = None) -> bool:
        """Create a new branch."""
        if branch_name in self.branches:
            logger.error(f"Branch {branch_name} already exists")
            return False
        
        base_commit = base_commit or self.head
        
        if not base_commit:
            logger.error("No base commit specified and no HEAD commit")
            return False
        
        branch = ModelBranch(
            branch_name=branch_name,
            created_at=datetime.now(),
            created_by=author,
            base_commit_id=base_commit,
            current_commit_id=base_commit,
            description=description
        )
        
        self.branches[branch_name] = branch
        self._save_branches(self.branches)
        
        logger.info(f"Created branch {branch_name} at {base_commit}")
        return True
    
    def merge(self, source_branch: str, target_branch: str = None,
             author: str = None, message: str = None) -> Optional[str]:
        """
        Merge branches (simplified - takes source branch head).
        
        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into (default: current)
            author: Merge author
            message: Merge commit message
            
        Returns:
            Merge commit ID if successful
        """
        target_branch = target_branch or self.current_branch
        
        if source_branch not in self.branches:
            logger.error(f"Source branch {source_branch} not found")
            return None
        
        if target_branch not in self.branches:
            logger.error(f"Target branch {target_branch} not found")
            return None
        
        source_commit = self.branches[source_branch].current_commit_id
        
        # Load source model and data
        source_model = self.load_model(source_commit)
        source_data = self._load_commit_data(source_commit)
        
        if not source_model or not source_data:
            logger.error("Failed to load source commit data")
            return None
        
        # Create merge commit
        merge_message = message or f"Merge branch '{source_branch}' into {target_branch}"
        
        merge_commit_id = self.commit(
            model=source_model,
            model_id=self.commits[source_commit].model_id,
            author=author or "system",
            message=merge_message,
            parameters=source_data['parameters'],
            architecture=source_data['architecture'],
            performance=source_data['performance'],
            tags=["merge"]
        )
        
        logger.info(f"Merged {source_branch} into {target_branch}: {merge_commit_id}")
        return merge_commit_id
    
    def tag(self, tag_name: str, commit_id: Optional[str] = None):
        """Create a tag for a commit."""
        commit_id = commit_id or self.head
        
        if not commit_id or commit_id not in self.commits:
            logger.error("Invalid commit ID")
            return
        
        self.tags[tag_name] = commit_id
        self._save_tags()
        
        logger.info(f"Tagged {commit_id} as {tag_name}")
    
    def diff(self, from_commit: str, to_commit: str) -> ModelDiff:
        """
        Generate diff between two commits.
        
        Args:
            from_commit: Source commit ID
            to_commit: Target commit ID
            
        Returns:
            ModelDiff object
        """
        # Load commit data
        from_data = self._load_commit_data(from_commit)
        to_data = self._load_commit_data(to_commit)
        
        if not from_data or not to_data:
            raise ValueError("Failed to load commit data")
        
        # Calculate parameter differences
        param_diff = self._detailed_diff(
            from_data.get('parameters', {}),
            to_data.get('parameters', {})
        )
        
        # Calculate architecture differences
        arch_diff = self._detailed_diff(
            from_data.get('architecture', {}),
            to_data.get('architecture', {})
        )
        
        # Calculate performance differences
        perf_changes = {}
        from_perf = from_data.get('performance', {})
        to_perf = to_data.get('performance', {})
        
        all_metrics = set(from_perf.keys()) | set(to_perf.keys())
        for metric in all_metrics:
            old_val = from_perf.get(metric, 0)
            new_val = to_perf.get(metric, 0)
            delta = new_val - old_val
            perf_changes[metric] = (old_val, new_val, delta)
        
        # Generate summary
        summary_parts = []
        if param_diff['added']:
            summary_parts.append(f"{len(param_diff['added'])} parameters added")
        if param_diff['removed']:
            summary_parts.append(f"{len(param_diff['removed'])} parameters removed")
        if param_diff['modified']:
            summary_parts.append(f"{len(param_diff['modified'])} parameters modified")
        
        # Performance summary
        improvements = sum(1 for _, _, delta in perf_changes.values() if delta > 0)
        regressions = sum(1 for _, _, delta in perf_changes.values() if delta < 0)
        if improvements:
            summary_parts.append(f"{improvements} metrics improved")
        if regressions:
            summary_parts.append(f"{regressions} metrics regressed")
        
        summary = "; ".join(summary_parts) if summary_parts else "No changes"
        
        return ModelDiff(
            from_commit=from_commit,
            to_commit=to_commit,
            param_added=param_diff['added'],
            param_removed=param_diff['removed'],
            param_modified=param_diff['modified'],
            arch_added=arch_diff['added'],
            arch_removed=arch_diff['removed'],
            arch_modified=arch_diff['modified'],
            perf_changes=perf_changes,
            summary=summary
        )
    
    def _detailed_diff(self, old_dict: Dict[str, Any], 
                      new_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed differences between dictionaries."""
        diff = {
            'added': {},
            'removed': {},
            'modified': {}
        }
        
        # Find added keys
        for key in new_dict:
            if key not in old_dict:
                diff['added'][key] = new_dict[key]
        
        # Find removed keys
        for key in old_dict:
            if key not in new_dict:
                diff['removed'][key] = old_dict[key]
        
        # Find modified values
        for key in old_dict:
            if key in new_dict:
                if isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                    # Recursive diff for nested dicts
                    nested_diff = self._detailed_diff(old_dict[key], new_dict[key])
                    if any(nested_diff.values()):
                        diff['modified'][key] = nested_diff
                elif old_dict[key] != new_dict[key]:
                    diff['modified'][key] = (old_dict[key], new_dict[key])
        
        return diff
    
    def load_model(self, commit_id: str) -> Optional[Any]:
        """Load model from a specific commit."""
        model_path = self.objects_path / f"{commit_id}_model.pkl"
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        return None
    
    def get_history(self, branch: Optional[str] = None, 
                   limit: int = 10) -> List[ModelCommit]:
        """Get commit history for a branch."""
        branch = branch or self.current_branch
        
        if branch not in self.branches:
            return []
        
        history = []
        current_commit_id = self.branches[branch].current_commit_id
        
        while current_commit_id and len(history) < limit:
            if current_commit_id in self.commits:
                commit = self.commits[current_commit_id]
                history.append(commit)
                current_commit_id = commit.parent_commit_id
            else:
                break
        
        return history
    
    def rollback(self, commit_id: str) -> bool:
        """Rollback to a specific commit."""
        if commit_id not in self.commits:
            logger.error(f"Commit {commit_id} not found")
            return False
        
        # Update current branch HEAD
        if self.current_branch in self.branches:
            self.branches[self.current_branch].current_commit_id = commit_id
            self._save_branches(self.branches)
        
        self.head = commit_id
        
        logger.info(f"Rolled back to commit {commit_id}")
        return True
    
    def get_performance_history(self, metric: str, 
                              branch: Optional[str] = None) -> pd.DataFrame:
        """Get performance history for a specific metric."""
        history = self.get_history(branch, limit=100)
        
        data = []
        for commit in history:
            commit_data = self._load_commit_data(commit.commit_id)
            if commit_data and 'performance' in commit_data:
                value = commit_data['performance'].get(metric, np.nan)
                data.append({
                    'commit_id': commit.commit_id,
                    'timestamp': commit.timestamp,
                    'author': commit.author,
                    'message': commit.message,
                    metric: value
                })
        
        return pd.DataFrame(data)