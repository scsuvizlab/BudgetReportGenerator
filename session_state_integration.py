"""
Session State Management with LLM Integration

Manages the current session state including loaded template, budget data,
field mappings, user configurations, and LLM functionality.
"""
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import json
import logging
from datetime import datetime

from template_document import TemplateDocument, Placeholder
from budget_book import BudgetBook, BudgetCell
from llm_integration_manager import LLMIntegrationManager, EnhancedMapping
from field_detector import FieldSuggestion

logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """Represents a mapping between a template field and budget data."""
    placeholder: Placeholder
    budget_cell: Optional[BudgetCell]
    manual_value: Optional[float]
    is_manually_set: bool = False
    confidence: float = 0.0
    notes: str = ""
    
    @property
    def final_value(self) -> Optional[float]:
        """Get the final value to use (manual override takes precedence)."""
        if self.is_manually_set and self.manual_value is not None:
            return self.manual_value
        elif self.budget_cell is not None:
            return self.budget_cell.value
        else:
            return None
    
    @property
    def display_value(self) -> str:
        """Get a formatted display value."""
        value = self.final_value
        if value is not None:
            return f"${value:,.2f}"
        else:
            return "No value"


@dataclass
class SessionConfig:
    """Configuration settings for the session."""
    openai_api_key: str = ""
    default_model: str = "gpt-4o-mini"
    cost_limit_usd: float = 5.0
    auto_save_enabled: bool = True
    output_format: str = "docx"  # "docx" or "md"
    output_directory: Path = field(default_factory=lambda: Path.home() / "Documents")
    auto_improve_enabled: bool = True
    heuristics_first: bool = True


class SessionState:
    """Manages the current session state and operations with LLM integration."""
    
    def __init__(self):
        self.config = SessionConfig()
        self.template: Optional[TemplateDocument] = None
        self.budget: Optional[BudgetBook] = None
        self.mappings: Dict[str, FieldMapping] = {}
        self.session_id = self._generate_session_id()
        self.is_dirty = False  # True if there are unsaved changes
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        
        # LLM Integration
        self.llm_manager = LLMIntegrationManager()
        self.llm_enabled = False
        self.field_suggestions: Dict[str, FieldSuggestion] = {}
        self.enhanced_mappings: Dict[str, EnhancedMapping] = {}
        
        # Session file paths
        self._session_dir = Path.home() / ".budget_tool" / "sessions"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"New session created: {self.session_id}")
    
    def initialize_llm(self, api_key: str, cost_limit: float = None) -> bool:
        """
        Initialize LLM integration with API key.
        
        Args:
            api_key: OpenAI API key
            cost_limit: Cost limit in USD (uses config default if None)
            
        Returns:
            True if initialization successful
        """
        if not api_key or not api_key.strip():
            logger.warning("No API key provided for LLM initialization")
            return False
        
        cost_limit = cost_limit or self.config.cost_limit_usd
        
        try:
            success = self.llm_manager.initialize_with_api_key(
                api_key=api_key.strip(),
                budget_limit=cost_limit,
                default_model=self.config.default_model
            )
            
            if success:
                self.llm_enabled = True
                self.config.openai_api_key = api_key.strip()
                self.config.cost_limit_usd = cost_limit
                logger.info("LLM integration enabled successfully")
                return True
            else:
                logger.error("Failed to initialize LLM integration")
                return False
                
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            return False
    
    def load_template(self, template_path: Path) -> bool:
        """Load a template document with LLM analysis."""
        try:
            from template_document import TemplateParser
            
            parser = TemplateParser()
            self.template = parser.parse_file(template_path)
            
            # Clear existing mappings when new template is loaded
            self.mappings.clear()
            self.field_suggestions.clear()
            self.enhanced_mappings.clear()
            
            # Create initial mappings for all placeholders
            for placeholder in self.template.placeholders:
                self.mappings[placeholder.text] = FieldMapping(
                    placeholder=placeholder,
                    budget_cell=None,
                    manual_value=None
                )
            
            # Analyze template with LLM if enabled
            if self.llm_enabled:
                try:
                    suggestions = self.llm_manager.analyze_template_fields(self.template)
                    
                    # Store suggestions
                    for suggestion in suggestions:
                        self.field_suggestions[suggestion.field_name] = suggestion
                    
                    logger.info(f"LLM template analysis complete: {len(suggestions)} suggestions")
                    
                except Exception as e:
                    logger.error(f"LLM template analysis failed: {e}")
                    # Continue without LLM analysis
            
            self.is_dirty = True
            logger.info(f"Template loaded: {template_path}")
            logger.info(f"Found {len(self.template.placeholders)} placeholders")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load template {template_path}: {e}")
            return False
    
    def load_budget(self, budget_path: Path) -> bool:
        """Load a budget spreadsheet."""
        try:
            from budget_book import BudgetParser
            
            parser = BudgetParser()
            self.budget = parser.parse_file(budget_path)
            
            # Try to auto-match existing mappings
            if self.template:
                self.enhanced_auto_match_fields()
            
            self.is_dirty = True
            logger.info(f"Budget loaded: {budget_path}")
            logger.info(f"Found {len(self.budget.cells)} budget cells")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load budget {budget_path}: {e}")
            return False
    
    def enhanced_auto_match_fields(self) -> None:
        """Enhanced field matching using both heuristics and LLM."""
        if not self.template or not self.budget:
            return
        
        if self.llm_enabled:
            try:
                # Use enhanced LLM-powered matching
                enhanced_mappings = self.llm_manager.enhanced_field_matching(
                    self.template, 
                    self.budget,
                    use_heuristics_first=self.config.heuristics_first
                )
                
                # Update existing mappings with enhanced results
                for field_name, enhanced_mapping in enhanced_mappings.items():
                    if field_name in self.mappings:
                        # Update existing FieldMapping object
                        mapping = self.mappings[field_name]
                        mapping.budget_cell = enhanced_mapping.budget_cell
                        mapping.confidence = enhanced_mapping.confidence
                        
                        # Add LLM-specific notes
                        if enhanced_mapping.source == 'llm':
                            mapping.notes = f"LLM: {enhanced_mapping.reasoning}"
                        elif enhanced_mapping.source == 'heuristic':
                            mapping.notes = f"Heuristic: {enhanced_mapping.reasoning}"
                        else:
                            mapping.notes = enhanced_mapping.reasoning
                        
                        # Store enhanced mapping for reference
                        self.enhanced_mappings[field_name] = enhanced_mapping
                
                logger.info("Enhanced field matching completed")
                
            except Exception as e:
                logger.error(f"Enhanced field matching failed: {e}")
                # Fallback to original heuristic matching
                self._auto_match_fields()
        else:
            # Use original heuristic matching
            self._auto_match_fields()
    
    def _auto_match_fields(self) -> None:
        """Automatically match template fields to budget cells using heuristics."""
        if not self.template or not self.budget:
            return
        
        for field_name, mapping in self.mappings.items():
            # Find best matching budget cell
            matches = self.budget.find_by_label(field_name)
            
            if matches:
                best_match = matches[0]  # Highest confidence first
                mapping.budget_cell = best_match
                mapping.confidence = best_match.confidence
                mapping.notes = f"Heuristic match to '{best_match.label}'"
                logger.info(f"Auto-matched '{field_name}' to ${best_match.value:,.2f}")
            else:
                logger.warning(f"No match found for field '{field_name}'")
    
    def improve_low_confidence_mappings(self, confidence_threshold: float = 0.6) -> int:
        """
        Use LLM to improve mappings with low confidence.
        
        Args:
            confidence_threshold: Minimum confidence to trigger improvement
            
        Returns:
            Number of mappings improved
        """
        if not self.llm_enabled or not self.config.auto_improve_enabled:
            return 0
        
        try:
            improved_count = self.llm_manager.improve_low_confidence_mappings(
                self.mappings, 
                self.budget, 
                confidence_threshold
            )
            
            # Update mappings with improvements
            for field_name, enhanced_mapping in self.llm_manager.enhanced_mappings.items():
                if field_name in self.mappings and field_name not in self.enhanced_mappings:
                    mapping = self.mappings[field_name]
                    mapping.budget_cell = enhanced_mapping.budget_cell
                    mapping.confidence = enhanced_mapping.confidence
                    mapping.notes = f"LLM improved: {enhanced_mapping.reasoning}"
                    self.enhanced_mappings[field_name] = enhanced_mapping
            
            return improved_count
            
        except Exception as e:
            logger.error(f"LLM improvement failed: {e}")
            return 0
    
    def set_manual_value(self, field_name: str, value: float, notes: str = "") -> bool:
        """Set a manual value for a field."""
        if field_name not in self.mappings:
            logger.error(f"Field '{field_name}' not found in mappings")
            return False
        
        mapping = self.mappings[field_name]
        mapping.manual_value = value
        mapping.is_manually_set = True
        mapping.notes = notes
        mapping.confidence = 1.0  # Manual values have perfect confidence
        
        self.is_dirty = True
        logger.info(f"Manual value set for '{field_name}': ${value:,.2f}")
        return True
    
    def clear_manual_value(self, field_name: str) -> bool:
        """Clear manual value and revert to auto-detected value."""
        if field_name not in self.mappings:
            return False
        
        mapping = self.mappings[field_name]
        mapping.manual_value = None
        mapping.is_manually_set = False
        
        # Restore notes and confidence from enhanced mapping if available
        if field_name in self.enhanced_mappings:
            enhanced = self.enhanced_mappings[field_name]
            mapping.confidence = enhanced.confidence
            mapping.notes = enhanced.reasoning
        elif mapping.budget_cell:
            mapping.confidence = mapping.budget_cell.confidence
            mapping.notes = f"Auto-detected: {mapping.budget_cell.label}"
        else:
            mapping.confidence = 0.0
            mapping.notes = ""
        
        self.is_dirty = True
        logger.info(f"Manual value cleared for '{field_name}'")
        return True
    
    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get a summary of all field mappings with LLM metrics."""
        summary = {
            "total_fields": len(self.mappings),
            "mapped_fields": 0,
            "manual_overrides": 0,
            "high_confidence": 0,
            "low_confidence": 0,
            "unmapped": 0
        }
        
        for mapping in self.mappings.values():
            if mapping.final_value is not None:
                summary["mapped_fields"] += 1
                
                if mapping.is_manually_set:
                    summary["manual_overrides"] += 1
                
                if mapping.confidence > 0.8:
                    summary["high_confidence"] += 1
                elif mapping.confidence > 0.5:
                    pass  # Medium confidence, no specific counter
                else:
                    summary["low_confidence"] += 1
            else:
                summary["unmapped"] += 1
        
        # Add LLM-specific metrics
        if self.llm_enabled:
            llm_enhanced_count = len([
                mapping for mapping in self.enhanced_mappings.values()
                if mapping.source in ['llm', 'llm_improvement']
            ])
            
            total_llm_cost = sum(
                mapping.llm_cost for mapping in self.enhanced_mappings.values()
            )
            
            summary.update({
                'llm_enabled': True,
                'llm_enhanced_mappings': llm_enhanced_count,
                'llm_total_cost': total_llm_cost,
                'field_suggestions': len(self.field_suggestions)
            })
        else:
            summary['llm_enabled'] = False
        
        return summary
    
    def get_llm_usage_summary(self) -> Dict[str, Any]:
        """Get LLM usage and cost summary."""
        if not self.llm_enabled:
            return {
                'enabled': False,
                'total_cost': 0.0,
                'remaining_budget': 0.0,
                'field_suggestions': 0,
                'enhanced_mappings': 0
            }
        
        usage = self.llm_manager.get_usage_summary()
        usage.update({
            'field_suggestions': len(self.field_suggestions),
            'enhanced_mappings': len(self.enhanced_mappings)
        })
        
        return usage
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key without fully initializing LLM."""
        try:
            return self.llm_manager.api_key_manager.set_api_key(api_key, save_to_keyring=False) and \
                   self.llm_manager.api_key_manager.validate_current_key()
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def configure_llm_settings(self, 
                             auto_improve: bool = True,
                             heuristics_first: bool = True,
                             cost_limit: float = 5.0) -> None:
        """Configure LLM behavior settings."""
        self.config.auto_improve_enabled = auto_improve
        self.config.heuristics_first = heuristics_first
        self.config.cost_limit_usd = cost_limit
        
        if self.llm_enabled and self.llm_manager.cost_guard:
            self.llm_manager.cost_guard.update_budget_limit(cost_limit)
    
    def is_ready_for_generation(self) -> bool:
        """Check if we have enough data to generate a document."""
        if not self.template or not self.mappings:
            return False
        
        # Check if at least 50% of fields are mapped
        mapped_count = sum(1 for m in self.mappings.values() if m.final_value is not None)
        return mapped_count >= len(self.mappings) * 0.5
    
    def add_cost(self, tokens: int, cost_usd: float) -> None:
        """Add to the running cost total."""
        self.total_tokens_used += tokens
        self.total_cost_usd += cost_usd
        
        if self.total_cost_usd > self.config.cost_limit_usd:
            logger.warning(f"Cost limit exceeded: ${self.total_cost_usd:.4f} > ${self.config.cost_limit_usd}")
    
    def can_afford(self, estimated_cost: float) -> bool:
        """Check if we can afford an operation."""
        return (self.total_cost_usd + estimated_cost) <= self.config.cost_limit_usd
    
    def export_llm_analysis(self, output_path: Path) -> bool:
        """Export detailed LLM analysis to JSON file."""
        if not self.llm_enabled:
            return False
        
        try:
            analysis_data = {
                'export_timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'llm_usage': self.get_llm_usage_summary(),
                'field_suggestions': {
                    name: {
                        'description': suggestion.description,
                        'data_type': suggestion.data_type,
                        'expected_context': suggestion.expected_context,
                        'confidence': suggestion.confidence,
                        'priority': suggestion.priority
                    }
                    for name, suggestion in self.field_suggestions.items()
                },
                'enhanced_mappings': {
                    name: {
                        'source': mapping.source,
                        'confidence': mapping.confidence,
                        'reasoning': mapping.reasoning,
                        'llm_cost': mapping.llm_cost,
                        'value': mapping.budget_cell.value if mapping.budget_cell else None
                    }
                    for name, mapping in self.enhanced_mappings.items()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"LLM analysis exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export LLM analysis: {e}")
            return False
    
    def save_session(self, filename: Optional[str] = None) -> Path:
        """Save the current session to disk."""
        if filename is None:
            filename = f"session_{self.session_id}.json"
        
        session_file = self._session_dir / filename
        
        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "openai_api_key": self.config.openai_api_key if self.config.openai_api_key else "",
                "default_model": self.config.default_model,
                "cost_limit_usd": self.config.cost_limit_usd,
                "output_format": self.config.output_format,
                "output_directory": str(self.config.output_directory),
                "auto_improve_enabled": self.config.auto_improve_enabled,
                "heuristics_first": self.config.heuristics_first
            },
            "template_path": str(self.template.source_path) if self.template else None,
            "budget_path": str(self.budget.source_path) if self.budget else None,
            "llm_enabled": self.llm_enabled,
            "mappings": {},
            "stats": {
                "total_tokens": self.total_tokens_used,
                "total_cost": self.total_cost_usd
            }
        }
        
        # Save mapping data (excluding object references)
        for field_name, mapping in self.mappings.items():
            session_data["mappings"][field_name] = {
                "manual_value": mapping.manual_value,
                "is_manually_set": mapping.is_manually_set,
                "confidence": mapping.confidence,
                "notes": mapping.notes,
                "budget_cell_info": {
                    "sheet": mapping.budget_cell.sheet,
                    "row": mapping.budget_cell.row,
                    "col": mapping.budget_cell.col,
                    "value": mapping.budget_cell.value
                } if mapping.budget_cell else None
            }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.is_dirty = False
        logger.info(f"Session saved to {session_file}")
        return session_file
    
    def load_session(self, session_file: Path) -> bool:
        """Load a session from disk."""
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Load configuration
            config_data = session_data.get("config", {})
            self.config.default_model = config_data.get("default_model", "gpt-4o-mini")
            self.config.cost_limit_usd = config_data.get("cost_limit_usd", 5.0)
            self.config.output_format = config_data.get("output_format", "docx")
            self.config.auto_improve_enabled = config_data.get("auto_improve_enabled", True)
            self.config.heuristics_first = config_data.get("heuristics_first", True)
            
            if "output_directory" in config_data:
                self.config.output_directory = Path(config_data["output_directory"])
            
            # Load API key if present
            api_key = config_data.get("openai_api_key", "")
            if api_key and session_data.get("llm_enabled", False):
                self.initialize_llm(api_key, self.config.cost_limit_usd)
            
            # Load template and budget if paths exist
            template_path = session_data.get("template_path")
            budget_path = session_data.get("budget_path")
            
            if template_path and Path(template_path).exists():
                self.load_template(Path(template_path))
            
            if budget_path and Path(budget_path).exists():
                self.load_budget(Path(budget_path))
            
            # Restore mapping states
            mapping_data = session_data.get("mappings", {})
            for field_name, data in mapping_data.items():
                if field_name in self.mappings:
                    mapping = self.mappings[field_name]
                    mapping.manual_value = data.get("manual_value")
                    mapping.is_manually_set = data.get("is_manually_set", False)
                    mapping.confidence = data.get("confidence", 0.0)
                    mapping.notes = data.get("notes", "")
            
            # Restore stats
            stats = session_data.get("stats", {})
            self.total_tokens_used = stats.get("total_tokens", 0)
            self.total_cost_usd = stats.get("total_cost", 0.0)
            
            self.session_id = session_data.get("session_id", self._generate_session_id())
            self.is_dirty = False
            
            logger.info(f"Session loaded from {session_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session {session_file}: {e}")
            return False
    
    def shutdown_llm(self) -> None:
        """Shutdown LLM integration and cleanup resources."""
        if self.llm_enabled:
            self.llm_manager.shutdown()
            self.llm_enabled = False
            self.field_suggestions.clear()
            self.enhanced_mappings.clear()
            logger.info("LLM integration shutdown")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def reset(self) -> None:
        """Reset the session to initial state."""
        # Shutdown LLM first
        self.shutdown_llm()
        
        self.template = None
        self.budget = None
        self.mappings.clear()
        self.field_suggestions.clear()
        self.enhanced_mappings.clear()
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.is_dirty = False
        self.session_id = self._generate_session_id()
        logger.info("Session reset")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    session = SessionState()
    
    # Test session functionality
    print(f"Session ID: {session.session_id}")
    print(f"LLM enabled: {session.llm_enabled}")
    print(f"Ready for generation: {session.is_ready_for_generation()}")
    
    # Test LLM configuration
    print(f"Can validate API key: {hasattr(session, 'validate_api_key')}")
    
    # Test mapping summary
    summary = session.get_mapping_summary()
    print(f"Mapping summary: {summary}")
    
    # Test LLM usage summary
    llm_usage = session.get_llm_usage_summary()
    print(f"LLM usage: {llm_usage}")