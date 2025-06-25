"""
Cell Resolver

Uses LLM to resolve template fields to specific budget cells when heuristic matching fails.
"""
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from llm_client import LLMClient
from budget_book import BudgetCell, BudgetBook
from field_detector import FieldSuggestion

logger = logging.getLogger(__name__)


@dataclass
class CellResolution:
    """Result of LLM cell value resolution."""
    field_name: str
    resolved_value: Optional[float]
    confidence: float
    reasoning: str
    source_cells: List[str]  # Cell references like "B8", "Sheet1!C10"
    calculation_method: str  # 'direct', 'sum', 'calculation'
    
    @property
    def success(self) -> bool:
        """Whether resolution was successful."""
        return self.resolved_value is not None and self.confidence > 0.5


class CellResolver:
    """
    Resolves template fields to budget cells using LLM analysis.
    
    Responsibilities:
    - Analyze budget spreadsheet context around candidate cells
    - Use LLM to determine best cell matches
    - Handle complex calculations (sum, formulas)
    - Provide reasoning for matches
    """
    
    CELL_RESOLUTION_PROMPT = """You are analyzing a budget spreadsheet to find the correct value for a template field.

FIELD TO RESOLVE:
Field: {field_name}
Field Description: {field_description}
Expected Data Type: {data_type}

BUDGET CONTEXT:
{budget_context}

CANDIDATE CELLS:
{candidate_cells}

TASK:
Determine the correct value for this field by analyzing the candidate cells and their context.

Consider:
1. Label/description similarity to the field requirement
2. Value magnitude (reasonable for this type of field)
3. Cell location and surrounding context
4. Whether multiple cells need to be combined (summed)

If multiple cells should be combined, identify all relevant cells.
If no candidates are appropriate, return null for the value.

Return your analysis as JSON:
{{
  "resolved_value": 75000.50,
  "confidence": 0.85,
  "reasoning": "Selected Cell 2 because 'PI Total Compensation' directly matches the field requirement. The value $75,000 is reasonable for principal investigator total cost including fringe.",
  "source_cells": ["B8"],
  "calculation_method": "direct"
}}

Calculation methods:
- "direct": Single cell value
- "sum": Sum of multiple cells  
- "calculation": More complex calculation

For currency values, return only the final numeric amount."""
    
    BUDGET_CONTEXT_PROMPT = """Analyze this budget spreadsheet structure and provide context for field resolution.

BUDGET INFORMATION:
Sheets: {sheet_names}
Total cells: {total_cells}
Years found: {years}

SAMPLE DATA:
{sample_data}

Provide a 2-3 sentence summary of the budget structure, main categories, and data organization."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize cell resolver.
        
        Args:
            llm_client: Configured LLM client for analysis
        """
        self.llm_client = llm_client
        self.budget_context_cache: Dict[str, str] = {}
        self.resolution_cost = 0.0
    
    def resolve_field_to_cells(self, 
                             field_name: str,
                             field_suggestion: Optional[FieldSuggestion],
                             candidate_cells: List[BudgetCell],
                             budget_book: BudgetBook) -> CellResolution:
        """
        Resolve a template field to specific budget cell(s).
        
        Args:
            field_name: Name of the template field
            field_suggestion: LLM suggestion for this field (if available)
            candidate_cells: List of potential matching cells
            budget_book: Full budget book for context
            
        Returns:
            Cell resolution result
        """
        if not candidate_cells:
            logger.warning(f"No candidate cells provided for field {field_name}")
            return CellResolution(
                field_name=field_name,
                resolved_value=None,
                confidence=0.0,
                reasoning="No candidate cells available",
                source_cells=[],
                calculation_method="none"
            )
        
        try:
            # Get budget context
            budget_context = self._get_budget_context(budget_book)
            
            # Prepare field information
            field_description = field_suggestion.description if field_suggestion else "Budget value needed"
            data_type = field_suggestion.data_type if field_suggestion else "currency"
            
            # Format candidate cells for LLM
            candidate_text = self._format_candidate_cells(candidate_cells)
            
            # Create prompt
            prompt = self.CELL_RESOLUTION_PROMPT.format(
                field_name=field_name,
                field_description=field_description,
                data_type=data_type,
                budget_context=budget_context,
                candidate_cells=candidate_text
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a budget analysis expert with deep knowledge of academic grant budgets and spreadsheet structures."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Make LLM call
            response = self.llm_client.call(
                messages=messages,
                max_tokens=400,
                temperature=0.1
            )
            
            self.resolution_cost += response.cost_usd
            
            if not response.success:
                logger.error(f"LLM cell resolution failed: {response.error_message}")
                return self._create_failed_resolution(field_name, "LLM call failed")
            
            # Parse response
            resolution = self._parse_resolution_response(field_name, response.content, candidate_cells)
            
            logger.info(f"Cell resolution for {field_name}: {resolution.reasoning}")
            return resolution
            
        except Exception as e:
            logger.error(f"Cell resolution failed for {field_name}: {e}")
            return self._create_failed_resolution(field_name, f"Error: {str(e)}")
    
    def improve_low_confidence_resolution(self,
                                        field_name: str,
                                        current_cell: BudgetCell,
                                        budget_book: BudgetBook,
                                        confidence_threshold: float = 0.6) -> Optional[CellResolution]:
        """
        Try to improve a low-confidence cell resolution.
        
        Args:
            field_name: Name of the field
            current_cell: Currently matched cell
            budget_book: Budget book for additional context
            confidence_threshold: Minimum confidence to attempt improvement
            
        Returns:
            Improved resolution if possible, None otherwise
        """
        if not current_cell or current_cell.confidence > confidence_threshold:
            return None
        
        try:
            # Get broader set of candidates
            all_candidates = budget_book.find_by_label(field_name)
            
            # Add cells with similar values
            value_tolerance = current_cell.value * 0.1  # 10% tolerance
            similar_value_cells = [
                cell for cell in budget_book.cells
                if abs(cell.value - current_cell.value) <= value_tolerance
            ]
            
            # Combine and deduplicate candidates
            candidates = list({cell.row: cell for cell in (all_candidates + similar_value_cells)}.values())
            candidates = candidates[:10]  # Limit to avoid token costs
            
            # Use existing resolution logic
            resolution = self.resolve_field_to_cells(
                field_name=field_name,
                field_suggestion=None,
                candidate_cells=candidates,
                budget_book=budget_book
            )
            
            if resolution.success and resolution.confidence > current_cell.confidence:
                logger.info(f"Improved resolution for {field_name}: {current_cell.confidence:.2f} → {resolution.confidence:.2f}")
                return resolution
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to improve resolution for {field_name}: {e}")
            return None
    
    def _get_budget_context(self, budget_book: BudgetBook) -> str:
        """
        Get or generate budget context description.
        
        Args:
            budget_book: Budget book to analyze
            
        Returns:
            Context description string
        """
        # Use cache if available
        cache_key = f"{budget_book.source_path}_{len(budget_book.cells)}"
        if cache_key in self.budget_context_cache:
            return self.budget_context_cache[cache_key]
        
        try:
            # Create sample data for context
            sample_cells = sorted(budget_book.cells, key=lambda x: x.value, reverse=True)[:10]
            sample_data = []
            
            for cell in sample_cells:
                sample_data.append(f"  {cell.label}: ${cell.value:,.2f} (Sheet: {cell.sheet})")
            
            sample_text = "\n".join(sample_data)
            
            # Generate context with LLM
            prompt = self.BUDGET_CONTEXT_PROMPT.format(
                sheet_names=", ".join(budget_book.sheets),
                total_cells=len(budget_book.cells),
                years=budget_book.get_years(),
                sample_data=sample_text
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a budget analyst. Provide concise summaries of budget structure."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = self.llm_client.call(messages, max_tokens=200, temperature=0.1)
            
            if response.success:
                context = response.content.strip()
                self.budget_context_cache[cache_key] = context
                return context
        
        except Exception as e:
            logger.warning(f"Failed to generate budget context: {e}")
        
        # Fallback to simple context
        context = f"Budget with {len(budget_book.sheets)} sheets, {len(budget_book.cells)} total cells, years: {budget_book.get_years()}"
        self.budget_context_cache[cache_key] = context
        return context
    
    def _format_candidate_cells(self, cells: List[BudgetCell]) -> str:
        """
        Format candidate cells for LLM analysis.
        
        Args:
            cells: List of candidate cells
            
        Returns:
            Formatted text description of candidates
        """
        if not cells:
            return "No candidate cells available."
        
        formatted = []
        for i, cell in enumerate(cells[:15], 1):  # Limit to 15 to avoid token limits
            location = f"{cell.sheet}!{self._get_cell_reference(cell.row, cell.col)}"
            year_info = f", Year: {cell.year}" if cell.year else ""
            
            formatted.append(
                f"Cell {i}: '{cell.label}' = ${cell.value:,.2f} "
                f"(Location: {location}, Confidence: {cell.confidence:.2f}{year_info})"
            )
        
        if len(cells) > 15:
            formatted.append(f"... and {len(cells) - 15} more candidates")
        
        return "\n".join(formatted)
    
    def _get_cell_reference(self, row: int, col: int) -> str:
        """
        Convert row/column numbers to Excel cell reference.
        
        Args:
            row: Row number (1-based)
            col: Column number (1-based)
            
        Returns:
            Excel cell reference like "B8"
        """
        # Convert column number to letters
        letters = ""
        while col > 0:
            col -= 1
            letters = chr(65 + (col % 26)) + letters
            col //= 26
        
        return f"{letters}{row}"
    
    def _parse_resolution_response(self, 
                                 field_name: str, 
                                 llm_response: str, 
                                 candidate_cells: List[BudgetCell]) -> CellResolution:
        """
        Parse LLM response into CellResolution.
        
        Args:
            field_name: Name of the field being resolved
            llm_response: Raw LLM response
            candidate_cells: Original candidate cells
            
        Returns:
            Parsed resolution result
        """
        try:
            # Extract JSON from response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON found in response")
            
            json_content = llm_response[json_start:json_end]
            data = json.loads(json_content)
            
            # Extract values with defaults
            resolved_value = data.get('resolved_value')
            if resolved_value is not None:
                resolved_value = float(resolved_value)
            
            confidence = float(data.get('confidence', 0.0))
            reasoning = data.get('reasoning', 'No reasoning provided')
            source_cells = data.get('source_cells', [])
            calculation_method = data.get('calculation_method', 'unknown')
            
            return CellResolution(
                field_name=field_name,
                resolved_value=resolved_value,
                confidence=confidence,
                reasoning=reasoning,
                source_cells=source_cells,
                calculation_method=calculation_method
            )
            
        except Exception as e:
            logger.error(f"Failed to parse resolution response: {e}")
            logger.debug(f"Raw response: {llm_response}")
            
            return self._create_failed_resolution(field_name, f"Parse error: {str(e)}")
    
    def _create_failed_resolution(self, field_name: str, reason: str) -> CellResolution:
        """
        Create a failed resolution result.
        
        Args:
            field_name: Name of the field
            reason: Reason for failure
            
        Returns:
            Failed resolution result
        """
        return CellResolution(
            field_name=field_name,
            resolved_value=None,
            confidence=0.0,
            reasoning=f"Resolution failed: {reason}",
            source_cells=[],
            calculation_method="failed"
        )
    
    def get_resolution_stats(self) -> Dict[str, Any]:
        """
        Get resolution statistics.
        
        Returns:
            Dictionary with resolution stats
        """
        return {
            'total_resolution_cost': self.resolution_cost,
            'cached_contexts': len(self.budget_context_cache),
            'llm_client_available': self.llm_client is not None
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Cell Resolver Example:")
    print("This component requires both LLM client and budget data to function.")
    print("\nExample usage:")
    print("""
    # Initialize with LLM client
    resolver = CellResolver(llm_client)
    
    # Resolve field to cells
    resolution = resolver.resolve_field_to_cells(
        field_name="{PI_Salary_Total}",
        field_suggestion=field_suggestion,
        candidate_cells=candidate_cells,
        budget_book=budget_book
    )
    
    if resolution.success:
        print(f"Resolved value: ${resolution.resolved_value:,.2f}")
        print(f"Confidence: {resolution.confidence:.2f}")
        print(f"Reasoning: {resolution.reasoning}")
    """)
    
    # Example of what LLM resolution might return
    example_resolution = {
        "resolved_value": 123456.78,
        "confidence": 0.92,
        "reasoning": "Cell 3 contains 'PI Total Compensation' which directly matches the required field. The value $123,456.78 includes both salary and fringe benefits as expected.",
        "source_cells": ["B12"],
        "calculation_method": "direct"
    }
    
    print(f"\nExample LLM resolution:")
    print(json.dumps(example_resolution, indent=2))
