"""
LLM Integration Manager

Orchestrates all LLM components and integrates with the session state.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from llm_client import LLMClient
from cost_guard import CostGuard
from api_key_manager import APIKeyManager
from field_detector import FieldDetector, FieldSuggestion
from cell_resolver import CellResolver, CellResolution
from template_document import TemplateDocument
from budget_book import BudgetBook, BudgetCell

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMapping:
    """Enhanced field mapping result combining heuristics and LLM."""
    field_name: str
    budget_cell: Optional[BudgetCell]
    confidence: float
    source: str  # 'heuristic', 'llm', 'hybrid', 'manual'
    reasoning: str
    llm_cost: float = 0.0
    field_suggestion: Optional[FieldSuggestion] = None
    cell_resolution: Optional[CellResolution] = None


class LLMIntegrationManager:
    """
    Manages all LLM integration functionality.
    
    Responsibilities:
    - Initialize and coordinate LLM components
    - Orchestrate enhanced field detection and resolution
    - Manage costs and budget limits
    - Provide unified interface for session state
    """
    
    def __init__(self):
        """Initialize LLM integration manager."""
        self.api_key_manager = APIKeyManager()
        self.cost_guard: Optional[CostGuard] = None
        self.llm_client: Optional[LLMClient] = None
        self.field_detector: Optional[FieldDetector] = None
        self.cell_resolver: Optional[CellResolver] = None
        
        self.field_suggestions: Dict[str, FieldSuggestion] = {}
        self.enhanced_mappings: Dict[str, EnhancedMapping] = {}
        self.is_initialized = False
    
    def initialize_with_api_key(self, 
                              api_key: str, 
                              budget_limit: float = 5.0,
                              default_model: str = "gpt-4o-mini") -> bool:
        """
        Initialize all LLM components with API key.
        
        Args:
            api_key: OpenAI API key
            budget_limit: Cost limit in USD
            default_model: Default LLM model to use
            
        Returns:
            True if initialization successful
        """
        try:
            # Store API key
            if not self.api_key_manager.set_api_key(api_key):
                logger.error("Failed to set API key")
                return False
            
            # Initialize cost guard
            self.cost_guard = CostGuard(budget_limit_usd=budget_limit)
            
            # Initialize LLM client
            self.llm_client = LLMClient(api_key, default_model)
            
            # Validate API key
            if not self.llm_client.validate_api_key():
                logger.error("API key validation failed")
                return False
            
            # Initialize specialized components
            self.field_detector = FieldDetector(self.llm_client)
            self.cell_resolver = CellResolver(self.llm_client)
            
            self.is_initialized = True
            logger.info("LLM integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            self.is_initialized = False
            return False
    
    def analyze_template_fields(self, template: TemplateDocument) -> List[FieldSuggestion]:
        """
        Analyze template to detect fields needing budget values.
        
        Args:
            template: Template document to analyze
            
        Returns:
            List of field suggestions from LLM
        """
        if not self._check_initialization():
            return []
        
        try:
            # Check cost before proceeding
            estimated_cost = 0.50  # Rough estimate for template analysis
            if not self.cost_guard.check_affordability(estimated_cost):
                logger.warning("Cannot afford template analysis - budget limit reached")
                return []
            
            # Analyze template
            suggestions = self.field_detector.analyze_template(template)
            
            # Record actual cost
            actual_cost = self.field_detector.last_analysis_cost
            self.cost_guard.record_cost(
                cost_usd=actual_cost,
                tokens=0,  # Tokens tracked in LLM client
                model=self.llm_client.default_model,
                operation="template_analysis"
            )
            
            # Store suggestions
            for suggestion in suggestions:
                self.field_suggestions[suggestion.field_name] = suggestion
            
            logger.info(f"Template analysis complete: {len(suggestions)} suggestions, cost ${actual_cost:.4f}")
            return suggestions
            
        except Exception as e:
            logger.error(f"Template analysis failed: {e}")
            return []
    
    def enhanced_field_matching(self, 
                              template: TemplateDocument, 
                              budget_book: BudgetBook,
                              use_heuristics_first: bool = True) -> Dict[str, EnhancedMapping]:
        """
        Perform enhanced field matching using heuristics + LLM.
        
        Args:
            template: Template document
            budget_book: Budget spreadsheet data
            use_heuristics_first: Whether to try heuristics before LLM
            
        Returns:
            Dictionary of enhanced mappings
        """
        if not self._check_initialization():
            return {}
        
        try:
            mappings = {}
            
            # Analyze template if not done yet
            if not self.field_suggestions:
                self.analyze_template_fields(template)
            
            for placeholder in template.placeholders:
                field_name = placeholder.text
                logger.info(f"Processing field: {field_name}")
                
                mapping = self._process_single_field(
                    field_name, 
                    placeholder, 
                    budget_book, 
                    use_heuristics_first
                )
                
                mappings[field_name] = mapping
                self.enhanced_mappings[field_name] = mapping
            
            return mappings
            
        except Exception as e:
            logger.error(f"Enhanced field matching failed: {e}")
            return {}
    
    def improve_low_confidence_mappings(self, 
                                      current_mappings: Dict[str, Any],
                                      budget_book: BudgetBook,
                                      confidence_threshold: float = 0.6) -> int:
        """
        Use LLM to improve mappings with low confidence.
        
        Args:
            current_mappings: Current field mappings
            budget_book: Budget data
            confidence_threshold: Threshold for improvement
            
        Returns:
            Number of mappings improved
        """
        if not self._check_initialization():
            return 0
        
        improved_count = 0
        
        for field_name, mapping in current_mappings.items():
            if (hasattr(mapping, 'confidence') and 
                mapping.confidence < confidence_threshold and
                not getattr(mapping, 'is_manually_set', False)):
                
                try:
                    # Check cost
                    estimated_cost = 0.20
                    if not self.cost_guard.check_affordability(estimated_cost):
                        logger.warning(f"Cannot afford to improve {field_name} - budget limit reached")
                        continue
                    
                    current_cell = getattr(mapping, 'budget_cell', None)
                    if current_cell:
                        resolution = self.cell_resolver.improve_low_confidence_resolution(
                            field_name, current_cell, budget_book, confidence_threshold
                        )
                        
                        if resolution and resolution.success and resolution.confidence > mapping.confidence:
                            # Update mapping with improved resolution
                            enhanced_mapping = EnhancedMapping(
                                field_name=field_name,
                                budget_cell=self._find_cell_by_value(budget_book, resolution.resolved_value),
                                confidence=resolution.confidence,
                                source='llm_improvement',
                                reasoning=f"LLM improved: {resolution.reasoning}",
                                llm_cost=self.cell_resolver.resolution_cost,
                                cell_resolution=resolution
                            )
                            
                            self.enhanced_mappings[field_name] = enhanced_mapping
                            improved_count += 1
                            
                            logger.info(f"Improved {field_name}: {mapping.confidence:.2f} → {resolution.confidence:.2f}")
                
                except Exception as e:
                    logger.error(f"Failed to improve mapping for {field_name}: {e}")
        
        logger.info(f"LLM improved {improved_count} low-confidence mappings")
        return improved_count
    
    def _process_single_field(self, 
                            field_name: str, 
                            placeholder: Any,
                            budget_book: BudgetBook,
                            use_heuristics_first: bool) -> EnhancedMapping:
        """
        Process a single field using best available method.
        
        Args:
            field_name: Name of the field
            placeholder: Template placeholder object
            budget_book: Budget data
            use_heuristics_first: Whether to try heuristics first
            
        Returns:
            Enhanced mapping result
        """
        # Try heuristic matching first if requested
        if use_heuristics_first:
            heuristic_matches = budget_book.find_by_label(field_name)
            
            if heuristic_matches and heuristic_matches[0].confidence > 0.8:
                # High confidence heuristic match
                best_match = heuristic_matches[0]
                return EnhancedMapping(
                    field_name=field_name,
                    budget_cell=best_match,
                    confidence=best_match.confidence,
                    source='heuristic',
                    reasoning=f"High confidence heuristic match to '{best_match.label}'"
                )
        
        # Use LLM resolution
        try:
            # Check cost
            estimated_cost = 0.30
            if not self.cost_guard.check_affordability(estimated_cost):
                logger.warning(f"Cannot afford LLM resolution for {field_name}")
                return self._create_failed_mapping(field_name, "Budget limit reached")
            
            # Get candidates for LLM analysis
            candidates = budget_book.find_by_label(field_name)
            if not candidates:
                # Expand search if no direct matches
                clean_field = field_name.replace('{', '').replace('}', '').replace('_', ' ')
                candidates = budget_book.find_by_label(clean_field)
            
            if not candidates:
                # Use top values as candidates
                candidates = sorted(budget_book.cells, key=lambda x: x.value, reverse=True)[:15]
            
            # Get field suggestion
            field_suggestion = self.field_suggestions.get(field_name)
            
            # Resolve with LLM
            resolution = self.cell_resolver.resolve_field_to_cells(
                field_name, field_suggestion, candidates, budget_book
            )
            
            # Record cost
            self.cost_guard.record_cost(
                cost_usd=self.cell_resolver.resolution_cost,
                tokens=0,
                model=self.llm_client.default_model,
                operation="field_resolution"
            )
            
            if resolution.success:
                # Find the actual budget cell
                resolved_cell = self._find_cell_by_value(budget_book, resolution.resolved_value)
                
                return EnhancedMapping(
                    field_name=field_name,
                    budget_cell=resolved_cell,
                    confidence=resolution.confidence,
                    source='llm',
                    reasoning=resolution.reasoning,
                    llm_cost=self.cell_resolver.resolution_cost,
                    field_suggestion=field_suggestion,
                    cell_resolution=resolution
                )
            else:
                return self._create_failed_mapping(field_name, resolution.reasoning)
                
        except Exception as e:
            logger.error(f"LLM resolution failed for {field_name}: {e}")
            return self._create_failed_mapping(field_name, f"LLM error: {str(e)}")
    
    def _find_cell_by_value(self, budget_book: BudgetBook, value: Optional[float]) -> Optional[BudgetCell]:
        """
        Find a budget cell with the specified value.
        
        Args:
            budget_book: Budget data
            value: Value to find
            
        Returns:
            Matching cell if found
        """
        if value is None:
            return None
        
        # Look for exact match first
        for cell in budget_book.cells:
            if abs(cell.value - value) < 0.01:  # Allow small rounding differences
                return cell
        
        return None
    
    def _create_failed_mapping(self, field_name: str, reason: str) -> EnhancedMapping:
        """
        Create a failed mapping result.
        
        Args:
            field_name: Name of the field
            reason: Reason for failure
            
        Returns:
            Failed mapping
        """
        return EnhancedMapping(
            field_name=field_name,
            budget_cell=None,
            confidence=0.0,
            source='failed',
            reasoning=f"Failed: {reason}"
        )
    
    def _check_initialization(self) -> bool:
        """Check if LLM integration is properly initialized."""
        if not self.is_initialized:
            logger.warning("LLM integration not initialized")
            return False
        return True
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive usage summary.
        
        Returns:
            Dictionary with all usage statistics
        """
        summary = {
            'initialized': self.is_initialized,
            'api_key_status': self.api_key_manager.get_key_status() if self.api_key_manager else {},
            'total_cost': 0.0,
            'budget_remaining': 0.0,
            'enhanced_mappings': len(self.enhanced_mappings),
            'field_suggestions': len(self.field_suggestions)
        }
        
        if self.cost_guard:
            cost_breakdown = self.cost_guard.get_cost_breakdown()
            summary.update({
                'total_cost': cost_breakdown['total_cost'],
                'budget_remaining': cost_breakdown['budget_remaining'],
                'budget_utilization': cost_breakdown['budget_utilization'],
                'total_operations': cost_breakdown['total_operations']
            })
        
        if self.llm_client:
            llm_stats = self.llm_client.get_usage_stats()
            summary.update({
                'llm_tokens': llm_stats['total_tokens'],
                'llm_calls': llm_stats['call_count']
            })
        
        return summary
    
    def reset_session(self) -> None:
        """Reset session data but keep initialization."""
        self.field_suggestions.clear()
        self.enhanced_mappings.clear()
        
        if self.cost_guard:
            self.cost_guard.reset()
        
        logger.info("LLM session data reset")
    
    def shutdown(self) -> None:
        """Shutdown LLM integration and cleanup."""
        self.field_suggestions.clear()
        self.enhanced_mappings.clear()
        self.cost_guard = None
        self.llm_client = None
        self.field_detector = None
        self.cell_resolver = None
        self.is_initialized = False
        
        logger.info("LLM integration shutdown complete")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("LLM Integration Manager Example:")
    print("This component orchestrates all LLM functionality.")
    print("\nExample usage:")
    print("""
    # Initialize manager
    manager = LLMIntegrationManager()
    
    # Set up with API key
    success = manager.initialize_with_api_key("your-api-key", budget_limit=5.0)
    
    if success:
        # Analyze template
        suggestions = manager.analyze_template_fields(template)
        
        # Enhanced field matching
        mappings = manager.enhanced_field_matching(template, budget_book)
        
        # Improve low confidence mappings
        improved = manager.improve_low_confidence_mappings(mappings, budget_book)
        
        # Get usage summary
        usage = manager.get_usage_summary()
        print(f"Total cost: ${usage['total_cost']:.4f}")
    """)
