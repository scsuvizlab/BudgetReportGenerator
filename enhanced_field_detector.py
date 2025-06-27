"""
Enhanced Field Detector

Integrates LLM capabilities with existing heuristic field detection
to provide better field identification and budget cell resolution.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from template_document import TemplateDocument, Placeholder
from budget_book import BudgetBook, BudgetCell
from llm_client import LLMClient, FieldSuggestion, CellResolution
from session_state import FieldMapping

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMatch:
    """Result of enhanced field matching combining heuristics + LLM."""
    field_name: str
    budget_cell: Optional[BudgetCell]
    confidence: float
    source: str  # 'heuristic', 'llm', 'hybrid'
    reasoning: str
    llm_cost: float = 0.0


class EnhancedFieldDetector:
    """Enhanced field detection using both heuristics and LLM analysis."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.field_suggestions: Dict[str, FieldSuggestion] = {}
        self.budget_context: str = ""
    
    def analyze_template(self, template: TemplateDocument) -> List[FieldSuggestion]:
        """Analyze template using LLM to get better field understanding."""
        if not self.llm_client:
            logger.warning("No LLM client available for template analysis")
            return []
        
        try:
            suggestions = self.llm_client.analyze_template_fields(template.content)
            
            # Store suggestions for later use
            for suggestion in suggestions:
                self.field_suggestions[suggestion.field_name] = suggestion
            
            logger.info(f"LLM analyzed template and found {len(suggestions)} field suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"LLM template analysis failed: {e}")
            return []
    
    def generate_budget_context(self, budget: BudgetBook) -> str:
        """Generate contextual description of budget structure."""
        if not self.llm_client:
            return "Basic budget analysis"
        
        try:
            self.budget_context = self.llm_client.analyze_spreadsheet_context(budget)
            return self.budget_context
        except Exception as e:
            logger.error(f"Budget context generation failed: {e}")
            return "Budget context unavailable"
    
    def enhanced_field_matching(self, template: TemplateDocument, budget: BudgetBook) -> Dict[str, EnhancedMatch]:
        """Perform enhanced field matching using heuristics + LLM."""
        matches = {}
        
        # First, analyze template with LLM if available
        if self.llm_client:
            self.analyze_template(template)
            self.generate_budget_context(budget)
        
        # Process each placeholder
        for placeholder in template.placeholders:
            field_name = placeholder.text
            logger.info(f"Processing field: {field_name}")
            
            # Try heuristic matching first (existing logic)
            heuristic_matches = budget.find_by_label(field_name)
            
            if heuristic_matches and heuristic_matches[0].confidence > 0.8:
                # High confidence heuristic match - use it
                best_match = heuristic_matches[0]
                matches[field_name] = EnhancedMatch(
                    field_name=field_name,
                    budget_cell=best_match,
                    confidence=best_match.confidence,
                    source='heuristic',
                    reasoning=f"High confidence heuristic match to '{best_match.label}'"
                )
                logger.info(f"  ✓ Heuristic match: {best_match.label} = ${best_match.value:,.2f}")
            
            elif self.llm_client and field_name in self.field_suggestions:
                # Use LLM for complex resolution
                try:
                    llm_match = self._resolve_with_llm(field_name, placeholder, budget, heuristic_matches)
                    matches[field_name] = llm_match
                    logger.info(f"  🤖 LLM match: {llm_match.reasoning}")
                except Exception as e:
                    logger.error(f"  ✗ LLM resolution failed for {field_name}: {e}")
                    # Fallback to best heuristic match if available
                    if heuristic_matches:
                        best_match = heuristic_matches[0]
                        matches[field_name] = EnhancedMatch(
                            field_name=field_name,
                            budget_cell=best_match,
                            confidence=best_match.confidence * 0.7,  # Reduced confidence
                            source='heuristic_fallback',
                            reasoning=f"LLM failed, using heuristic match to '{best_match.label}'"
                        )
                    else:
                        matches[field_name] = EnhancedMatch(
                            field_name=field_name,
                            budget_cell=None,
                            confidence=0.0,
                            source='failed',
                            reasoning="No matches found and LLM resolution failed"
                        )
            
            elif heuristic_matches:
                # Use best heuristic match with reduced confidence
                best_match = heuristic_matches[0]
                matches[field_name] = EnhancedMatch(
                    field_name=field_name,
                    budget_cell=best_match,
                    confidence=best_match.confidence,
                    source='heuristic',
                    reasoning=f"Medium confidence heuristic match to '{best_match.label}'"
                )
                logger.info(f"  ≈ Heuristic match: {best_match.label} = ${best_match.value:,.2f}")
            
            else:
                # No matches found
                matches[field_name] = EnhancedMatch(
                    field_name=field_name,
                    budget_cell=None,
                    confidence=0.0,
                    source='none',
                    reasoning="No matching budget cells found"
                )
                logger.warning(f"  ✗ No match found for {field_name}")
        
        return matches
    
    def _resolve_with_llm(self, field_name: str, placeholder: Placeholder, budget: BudgetBook, candidates: List[BudgetCell]) -> EnhancedMatch:
        """Use LLM to resolve field to budget cell(s)."""
        
        # Get field suggestion if available
        field_suggestion = self.field_suggestions.get(field_name)
        field_context = field_suggestion.description if field_suggestion else placeholder.context
        
        # If no candidates from heuristics, do broader search
        if not candidates:
            # Try searching with cleaned field name
            clean_field = field_name.replace('{', '').replace('}', '').replace('_', ' ')
            candidates = budget.find_by_label(clean_field)
            
            # If still no candidates, get top budget items for LLM analysis
            if not candidates:
                candidates = sorted(budget.cells, key=lambda x: x.value, reverse=True)[:20]
        
        # Limit candidates to avoid token limits
        candidates = candidates[:15]
        
        # Call LLM for resolution
        resolution = self.llm_client.resolve_field_value(
            field_name=field_name,
            field_context=field_context,
            budget_context=self.budget_context,
            candidate_cells=candidates
        )
        
        # Find the budget cell that matches the resolution
        resolved_cell = None
        if resolution.value is not None:
            # Try to find the cell with matching value
            tolerance = 0.01  # Allow small rounding differences
            for cell in candidates:
                if abs(cell.value - resolution.value) <= tolerance:
                    resolved_cell = cell
                    break
            
            # If exact match not found, create a synthetic cell for calculated values
            if not resolved_cell and resolution.source_cells:
                # This indicates LLM calculated a value from multiple cells
                resolved_cell = BudgetCell(
                    sheet="calculated",
                    row=0,
                    col=0,
                    label=f"LLM calculated: {field_name}",
                    year=None,
                    value=resolution.value,
                    raw_value=resolution.value,
                    confidence=resolution.confidence,
                    context=f"Calculated from cells: {', '.join(resolution.source_cells)}"
                )
        
        return EnhancedMatch(
            field_name=field_name,
            budget_cell=resolved_cell,
            confidence=resolution.confidence,
            source='llm',
            reasoning=resolution.reasoning,
            llm_cost=self.llm_client.get_usage_summary()['total_cost'] if self.llm_client else 0.0
        )
    
    def improve_existing_mapping(self, field_name: str, current_mapping: FieldMapping, budget: BudgetBook) -> Optional[EnhancedMatch]:
        """Use LLM to improve an existing low-confidence mapping."""
        if not self.llm_client or current_mapping.confidence > 0.7:
            return None
        
        try:
            # Get broader candidate set
            candidates = budget.find_by_label(field_name)
            if not candidates:
                # Expand search
                clean_field = field_name.replace('{', '').replace('}', '').replace('_', ' ')
                candidates = budget.find_by_label(clean_field)
            
            if current_mapping.budget_cell:
                # Include current cell in candidates if not already there
                if current_mapping.budget_cell not in candidates:
                    candidates.insert(0, current_mapping.budget_cell)
            
            field_context = current_mapping.placeholder.context
            if field_name in self.field_suggestions:
                field_context = self.field_suggestions[field_name].description
            
            resolution = self.llm_client.resolve_field_value(
                field_name=field_name,
                field_context=field_context,
                budget_context=self.budget_context,
                candidate_cells=candidates[:15]
            )
            
            if resolution.confidence > current_mapping.confidence:
                logger.info(f"LLM improved mapping for {field_name}: {resolution.confidence:.2f} > {current_mapping.confidence:.2f}")
                
                # Find matching cell
                resolved_cell = None
                if resolution.value is not None:
                    for cell in candidates:
                        if abs(cell.value - resolution.value) <= 0.01:
                            resolved_cell = cell
                            break
                
                return EnhancedMatch(
                    field_name=field_name,
                    budget_cell=resolved_cell,
                    confidence=resolution.confidence,
                    source='llm_improvement',
                    reasoning=f"LLM improvement: {resolution.reasoning}",
                    llm_cost=self.llm_client.get_usage_summary()['total_cost']
                )
        
        except Exception as e:
            logger.error(f"LLM improvement failed for {field_name}: {e}")
        
        return None
    
    def get_unmapped_suggestions(self, unmapped_fields: List[str], budget: BudgetBook) -> Dict[str, List[BudgetCell]]:
        """Get LLM suggestions for completely unmapped fields."""
        if not self.llm_client or not unmapped_fields:
            return {}
        
        suggestions = {}
        
        for field_name in unmapped_fields:
            try:
                # Get top budget items for analysis
                top_cells = sorted(budget.cells, key=lambda x: x.value, reverse=True)[:20]
                
                field_context = "Unknown field requiring budget value"
                if field_name in self.field_suggestions:
                    field_context = self.field_suggestions[field_name].description
                
                resolution = self.llm_client.resolve_field_value(
                    field_name=field_name,
                    field_context=field_context,
                    budget_context=self.budget_context,
                    candidate_cells=top_cells
                )
                
                if resolution.value is not None and resolution.confidence > 0.5:
                    # Find matching cells
                    matching_cells = []
                    for cell in top_cells:
                        if abs(cell.value - resolution.value) <= 0.01:
                            matching_cells.append(cell)
                    
                    if matching_cells:
                        suggestions[field_name] = matching_cells
                        logger.info(f"LLM suggested matches for unmapped field {field_name}")
            
            except Exception as e:
                logger.error(f"Failed to get suggestions for {field_name}: {e}")
        
        return suggestions
    
    def get_detection_summary(self) -> Dict[str, any]:
        """Get summary of detection performance."""
        return {
            'llm_available': self.llm_client is not None,
            'field_suggestions': len(self.field_suggestions),
            'total_cost': self.llm_client.get_usage_summary()['total_cost'] if self.llm_client else 0.0,
            'budget_context_length': len(self.budget_context)
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Enhanced Field Detector implementation complete.")
    print("Integrate with session_state.py to enable LLM-powered field detection.")
