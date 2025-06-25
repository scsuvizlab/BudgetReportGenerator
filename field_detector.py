"""
Field Detector

Uses LLM to analyze budget justification templates and detect fields that need numeric values.
"""
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from llm_client import LLMClient, LLMResponse
from template_document import TemplateDocument

logger = logging.getLogger(__name__)


@dataclass
class FieldSuggestion:
    """Represents an LLM suggestion for a template field."""
    field_name: str
    description: str
    data_type: str  # 'currency', 'percentage', 'count', 'date', 'text'
    expected_context: str
    confidence: float
    priority: int = 1  # 1=high, 2=medium, 3=low


class FieldDetector:
    """
    Analyzes templates using LLM to detect and categorize placeholder fields.
    
    Responsibilities:
    - Analyze template content with LLM
    - Identify numeric placeholders
    - Categorize field types
    - Provide context for field matching
    """
    
    FIELD_ANALYSIS_PROMPT = """You are a grants administration expert analyzing a budget justification template.

Your task is to identify ALL placeholder fields that require numeric values from budget data.

Template content:
{template_content}

Instructions:
1. Find every placeholder field that needs a numeric value (currency, percentages, counts)
2. Ignore text fields like names, titles, descriptions
3. For each numeric field, determine:
   - The exact field name as it appears in the template
   - What type of budget data it represents
   - Where this data would likely be found in a budget spreadsheet

Return your analysis as a JSON array with this exact structure:
[
  {{
    "field_name": "{{PI_IFO_1_Total}}",
    "description": "Total cost for Principal Investigator including salary and fringe benefits",
    "data_type": "currency",
    "expected_context": "Principal investigator salary calculations with fringe benefits applied",
    "confidence": 0.95,
    "priority": 1
  }}
]

Data types:
- "currency": Dollar amounts like salaries, costs, totals
- "percentage": Rates like fringe rates, percentages
- "count": Counts like number of people, hours, days
- "date": Dates or years

Priority levels:
- 1: Critical fields that must be filled (totals, major costs)
- 2: Important but optional fields
- 3: Nice-to-have fields

Focus only on numeric fields that would come from budget calculations."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize field detector.
        
        Args:
            llm_client: Configured LLM client for analysis
        """
        self.llm_client = llm_client
        self.last_analysis_cost = 0.0
    
    def analyze_template(self, template: TemplateDocument) -> List[FieldSuggestion]:
        """
        Analyze template to detect fields that need numeric values.
        
        Args:
            template: Template document to analyze
            
        Returns:
            List of field suggestions from LLM analysis
        """
        if not template or not template.content:
            logger.warning("No template content to analyze")
            return []
        
        try:
            # Prepare the prompt
            prompt = self.FIELD_ANALYSIS_PROMPT.format(
                template_content=template.content[:4000]  # Limit content to avoid token limits
            )
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are a grants administration expert specializing in budget analysis and template processing."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Make LLM call
            response = self.llm_client.call(
                messages=messages,
                max_tokens=1500,
                temperature=0.1
            )
            
            self.last_analysis_cost = response.cost_usd
            
            if not response.success:
                logger.error(f"LLM field analysis failed: {response.error_message}")
                return []
            
            # Parse JSON response
            suggestions = self._parse_field_suggestions(response.content)
            
            logger.info(f"LLM field analysis complete: {len(suggestions)} suggestions, cost ${response.cost_usd:.4f}")
            return suggestions
            
        except Exception as e:
            logger.error(f"Field detection failed: {e}")
            return []
    
    def _parse_field_suggestions(self, llm_response: str) -> List[FieldSuggestion]:
        """
        Parse LLM response into FieldSuggestion objects.
        
        Args:
            llm_response: Raw response from LLM
            
        Returns:
            List of parsed field suggestions
        """
        suggestions = []
        
        try:
            # Try to extract JSON from response
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_content = llm_response[json_start:json_end]
                suggestions_data = json.loads(json_content)
                
                for item in suggestions_data:
                    # Validate required fields
                    if not all(key in item for key in ['field_name', 'description', 'data_type']):
                        logger.warning(f"Skipping incomplete field suggestion: {item}")
                        continue
                    
                    suggestion = FieldSuggestion(
                        field_name=item['field_name'],
                        description=item['description'],
                        data_type=item['data_type'],
                        expected_context=item.get('expected_context', ''),
                        confidence=float(item.get('confidence', 0.5)),
                        priority=int(item.get('priority', 2))
                    )
                    
                    suggestions.append(suggestion)
                    
            else:
                logger.warning("No JSON array found in LLM response")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {llm_response}")
            
        except Exception as e:
            logger.error(f"Error parsing field suggestions: {e}")
        
        return suggestions
    
    def enhance_existing_placeholders(self, 
                                    template: TemplateDocument, 
                                    suggestions: List[FieldSuggestion]) -> Dict[str, FieldSuggestion]:
        """
        Map LLM suggestions to existing template placeholders.
        
        Args:
            template: Template with existing placeholders
            suggestions: LLM field suggestions
            
        Returns:
            Dictionary mapping placeholder text to suggestions
        """
        mapping = {}
        
        # Create lookup for LLM suggestions
        suggestion_lookup = {
            self._normalize_field_name(s.field_name): s 
            for s in suggestions
        }
        
        # Match existing placeholders to suggestions
        for placeholder in template.placeholders:
            normalized_placeholder = self._normalize_field_name(placeholder.text)
            
            # Direct match
            if normalized_placeholder in suggestion_lookup:
                mapping[placeholder.text] = suggestion_lookup[normalized_placeholder]
                continue
            
            # Fuzzy match based on similarity
            best_match = None
            best_score = 0.0
            
            for suggestion in suggestions:
                score = self._calculate_similarity(
                    normalized_placeholder, 
                    self._normalize_field_name(suggestion.field_name)
                )
                
                if score > best_score and score > 0.7:  # Threshold for fuzzy matching
                    best_score = score
                    best_match = suggestion
            
            if best_match:
                mapping[placeholder.text] = best_match
                logger.info(f"Fuzzy matched '{placeholder.text}' to '{best_match.field_name}' (score: {best_score:.2f})")
        
        return mapping
    
    def get_unmapped_suggestions(self, 
                               suggestions: List[FieldSuggestion], 
                               mapped_fields: List[str]) -> List[FieldSuggestion]:
        """
        Get suggestions that weren't mapped to existing placeholders.
        
        Args:
            suggestions: All LLM suggestions
            mapped_fields: Field names that were successfully mapped
            
        Returns:
            List of unmapped suggestions
        """
        mapped_normalized = {self._normalize_field_name(field) for field in mapped_fields}
        
        unmapped = []
        for suggestion in suggestions:
            normalized_suggestion = self._normalize_field_name(suggestion.field_name)
            if normalized_suggestion not in mapped_normalized:
                unmapped.append(suggestion)
        
        # Sort by priority and confidence
        unmapped.sort(key=lambda s: (s.priority, -s.confidence))
        
        return unmapped
    
    def _normalize_field_name(self, field_name: str) -> str:
        """
        Normalize field name for comparison.
        
        Args:
            field_name: Raw field name
            
        Returns:
            Normalized field name
        """
        # Remove braces, brackets, and convert to lowercase
        normalized = field_name.lower()
        normalized = normalized.replace('{', '').replace('}', '')
        normalized = normalized.replace('[', '').replace(']', '')
        normalized = normalized.replace('_', ' ').replace('-', ' ')
        return normalized.strip()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of the last analysis performed.
        
        Returns:
            Analysis summary dictionary
        """
        return {
            'last_analysis_cost': self.last_analysis_cost,
            'llm_client_available': self.llm_client is not None,
            'supported_data_types': ['currency', 'percentage', 'count', 'date'],
            'priority_levels': {
                1: 'Critical fields (must be filled)',
                2: 'Important fields (should be filled)', 
                3: 'Optional fields (nice to have)'
            }
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Field Detector Example:")
    print("This component requires a valid LLM client to function.")
    print("\nExample usage:")
    print("""
    # Initialize with LLM client
    detector = FieldDetector(llm_client)
    
    # Analyze template
    suggestions = detector.analyze_template(template_doc)
    
    # Process suggestions
    for suggestion in suggestions:
        print(f"Field: {suggestion.field_name}")
        print(f"Type: {suggestion.data_type}")
        print(f"Description: {suggestion.description}")
        print(f"Confidence: {suggestion.confidence}")
    """)
    
    # Example of what LLM analysis might return
    example_suggestions = [
        {
            "field_name": "{PI_IFO_1_Total}",
            "description": "Total cost for Principal Investigator including salary and fringe",
            "data_type": "currency",
            "expected_context": "PI salary calculations with fringe benefits",
            "confidence": 0.95,
            "priority": 1
        },
        {
            "field_name": "{Travel_Domestic_Total}",
            "description": "Total domestic travel expenses",
            "data_type": "currency", 
            "expected_context": "Travel budget calculations for domestic trips",
            "confidence": 0.90,
            "priority": 2
        }
    ]
    
    print(f"\nExample LLM suggestions:")
    print(json.dumps(example_suggestions, indent=2))
