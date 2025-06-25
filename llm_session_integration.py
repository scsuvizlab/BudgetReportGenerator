"""
LLM Integration for Session State

Extends the existing session_state.py with LLM capabilities.
This should be merged into the existing session_state.py file.
"""
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

# These would be added to the existing session_state.py imports
from llm_client import LLMClient, FieldSuggestion
from enhanced_field_detector import EnhancedFieldDetector, EnhancedMatch

logger = logging.getLogger(__name__)


class SessionStateWithLLM:
    """Extended SessionState with LLM integration capabilities.
    
    This shows the additions that should be made to the existing SessionState class.
    """
    
    def __init__(self):
        # ... existing initialization code ...
        
        # New LLM-related attributes
        self.llm_client: Optional[LLMClient] = None
        self.field_detector: Optional[EnhancedFieldDetector] = None
        self.field_suggestions: Dict[str, FieldSuggestion] = {}
        self.llm_enhanced_mappings: Dict[str, EnhancedMatch] = {}
        self.llm_enabled = False
    
    def initialize_llm(self, api_key: str) -> bool:
        """Initialize LLM client with API key."""
        if not api_key or not api_key.strip():
            logger.warning("No API key provided for LLM initialization")
            return False
        
        try:
            self.llm_client = LLMClient(
                api_key=api_key.strip(),
                default_model=self.config.default_model
            )
            
            # Set cost limit
            self.llm_client.cost_guard.limit_usd = self.config.cost_limit_usd
            
            # Initialize enhanced field detector
            self.field_detector = EnhancedFieldDetector(self.llm_client)
            
            self.llm_enabled = True
            logger.info("LLM client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_enabled = False
            return False
    
    def analyze_template_with_llm(self) -> bool:
        """Analyze template using LLM for better field understanding."""
        if not self.llm_enabled or not self.template or not self.field_detector:
            return False
        
        try:
            suggestions = self.field_detector.analyze_template(self.template)
            
            # Store suggestions
            for suggestion in suggestions:
                self.field_suggestions[suggestion.field_name] = suggestion
            
            logger.info(f"LLM template analysis completed: {len(suggestions)} field suggestions")
            return True
            
        except Exception as e:
            logger.error(f"LLM template analysis failed: {e}")
            return False
    
    def enhanced_auto_match_fields(self) -> None:
        """Enhanced field matching using LLM + heuristics."""
        if not self.template or not self.budget:
            return
        
        if self.llm_enabled and self.field_detector:
            try:
                # Use enhanced field detector
                enhanced_matches = self.field_detector.enhanced_field_matching(
                    self.template, self.budget
                )
                
                # Update mappings with enhanced results
                for field_name, enhanced_match in enhanced_matches.items():
                    if field_name in self.mappings:
                        mapping = self.mappings[field_name]
                        
                        # Update with enhanced match
                        mapping.budget_cell = enhanced_match.budget_cell
                        mapping.confidence = enhanced_match.confidence
                        mapping.notes = f"Enhanced: {enhanced_match.reasoning}"
                        
                        # Store enhanced match details
                        self.llm_enhanced_mappings[field_name] = enhanced_match
                        
                        # Update cost tracking
                        if enhanced_match.llm_cost > 0:
                            self.add_cost(
                                tokens=0,  # Tokens already tracked in LLM client
                                cost_usd=enhanced_match.llm_cost
                            )
                
                logger.info("Enhanced field matching completed")
                
            except Exception as e:
                logger.error(f"Enhanced field matching failed: {e}")
                # Fallback to regular auto-matching
                self._auto_match_fields()
        else:
            # Use existing heuristic matching
            self._auto_match_fields()
    
    def improve_low_confidence_mappings(self, confidence_threshold: float = 0.6) -> int:
        """Use LLM to improve mappings with low confidence."""
        if not self.llm_enabled or not self.field_detector:
            return 0
        
        improved_count = 0
        
        for field_name, mapping in self.mappings.items():
            if mapping.confidence < confidence_threshold and not mapping.is_manually_set:
                try:
                    improved_match = self.field_detector.improve_existing_mapping(
                        field_name, mapping, self.budget
                    )
                    
                    if improved_match and improved_match.confidence > mapping.confidence:
                        # Update mapping with improvement
                        old_confidence = mapping.confidence
                        mapping.budget_cell = improved_match.budget_cell
                        mapping.confidence = improved_match.confidence
                        mapping.notes = f"LLM improved from {old_confidence:.2f}: {improved_match.reasoning}"
                        
                        self.llm_enhanced_mappings[field_name] = improved_match
                        improved_count += 1
                        
                        logger.info(f"Improved {field_name}: {old_confidence:.2f} → {improved_match.confidence:.2f}")
                
                except Exception as e:
                    logger.error(f"Failed to improve mapping for {field_name}: {e}")
        
        logger.info(f"LLM improved {improved_count} low-confidence mappings")
        return improved_count
    
    def get_unmapped_suggestions(self) -> Dict[str, List]:
        """Get LLM suggestions for completely unmapped fields."""
        if not self.llm_enabled or not self.field_detector:
            return {}
        
        # Find unmapped fields
        unmapped_fields = [
            field_name for field_name, mapping in self.mappings.items()
            if mapping.final_value is None and not mapping.is_manually_set
        ]
        
        if not unmapped_fields:
            return {}
        
        try:
            suggestions = self.field_detector.get_unmapped_suggestions(unmapped_fields, self.budget)
            logger.info(f"LLM provided suggestions for {len(suggestions)} unmapped fields")
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get unmapped suggestions: {e}")
            return {}
    
    def get_llm_usage_summary(self) -> Dict[str, Any]:
        """Get summary of LLM usage and costs."""
        if not self.llm_enabled or not self.llm_client:
            return {
                'enabled': False,
                'total_cost': 0.0,
                'remaining_budget': 0.0,
                'total_calls': 0
            }
        
        usage = self.llm_client.get_usage_summary()
        usage['enabled'] = True
        
        # Add field detection summary
        if self.field_detector:
            detection_summary = self.field_detector.get_detection_summary()
            usage.update(detection_summary)
        
        return usage
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key by making a minimal test call."""
        if not api_key or not api_key.strip():
            return False
        
        try:
            test_client = LLMClient(api_key.strip(), "gpt-4o-mini")
            
            # Make a minimal test call
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with just the word 'test'"}
            ]
            
            response = test_client.call_llm(messages, max_tokens=10)
            return "test" in response.content.lower()
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
    
    def export_llm_analysis(self, output_path: Path) -> bool:
        """Export detailed LLM analysis to JSON file."""
        if not self.llm_enabled:
            return False
        
        try:
            analysis_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'llm_usage': self.get_llm_usage_summary(),
                'field_suggestions': {
                    name: {
                        'description': suggestion.description,
                        'data_type': suggestion.data_type,
                        'expected_context': suggestion.expected_context,
                        'confidence': suggestion.confidence
                    }
                    for name, suggestion in self.field_suggestions.items()
                },
                'enhanced_matches': {
                    name: {
                        'confidence': match.confidence,
                        'source': match.source,
                        'reasoning': match.reasoning,
                        'value': match.budget_cell.value if match.budget_cell else None,
                        'llm_cost': match.llm_cost
                    }
                    for name, match in self.llm_enhanced_mappings.items()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"LLM analysis exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export LLM analysis: {e}")
            return False
    
    # Override existing methods to integrate LLM
    def load_template(self, template_path: Path) -> bool:
        """Enhanced template loading with LLM analysis."""
        # Call existing load_template logic
        success = super().load_template(template_path)  # This would be the existing method
        
        if success and self.llm_enabled:
            # Analyze template with LLM
            self.analyze_template_with_llm()
        
        return success
    
    def load_budget(self, budget_path: Path) -> bool:
        """Enhanced budget loading with LLM context generation."""
        # Call existing load_budget logic
        success = super().load_budget(budget_path)  # This would be the existing method
        
        if success and self.llm_enabled and self.field_detector:
            # Generate budget context for LLM
            try:
                context = self.field_detector.generate_budget_context(self.budget)
                logger.info("Budget context generated for LLM analysis")
            except Exception as e:
                logger.error(f"Failed to generate budget context: {e}")
        
        return success
    
    # Update existing get_mapping_summary to include LLM info
    def get_mapping_summary(self) -> Dict[str, Any]:
        """Enhanced mapping summary including LLM analysis."""
        # Get existing summary
        summary = super().get_mapping_summary()  # This would be the existing method
        
        if self.llm_enabled:
            # Add LLM-specific metrics
            llm_enhanced_count = len(self.llm_enhanced_mappings)
            llm_cost = sum(match.llm_cost for match in self.llm_enhanced_mappings.values())
            
            summary.update({
                'llm_enabled': True,
                'llm_enhanced_mappings': llm_enhanced_count,
                'llm_total_cost': llm_cost,
                'field_suggestions': len(self.field_suggestions)
            })
        else:
            summary['llm_enabled'] = False
        
        return summary


# Usage instructions for integration:
"""
TO INTEGRATE INTO EXISTING session_state.py:

1. Add the new imports at the top of session_state.py
2. Add the new attributes to SessionState.__init__()
3. Add all the new methods to the SessionState class
4. Modify the existing load_template() and load_budget() methods to call LLM analysis
5. Update get_mapping_summary() to include LLM metrics
6. Replace the call to _auto_match_fields() with enhanced_auto_match_fields()

The existing SessionConfig should also be updated to include:
- openai_api_key: str = ""
- default_model: str = "gpt-4o-mini" 
- cost_limit_usd: float = 5.0
"""
