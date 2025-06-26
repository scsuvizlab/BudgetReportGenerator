"""
Enhanced LLM Integration Manager - Orchestrates AI-powered analysis with grant-specific intelligence
Improved prompts and context awareness for better field detection and cell resolution
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Import enhanced components
from enhanced_budget_book import EnhancedBudgetBook, CellData
from enhanced_field_detector import EnhancedFieldDetector, DetectedField, FieldType
from enhanced_cell_resolver import EnhancedCellResolver, ResolutionResult, CellMatch

@dataclass
class LLMAnalysisRequest:
    """Request for LLM analysis"""
    template_text: str
    budget_summary: Dict[str, Any]
    analysis_type: str  # 'field_detection', 'cell_resolution', 'improvement_suggestions'
    context: Dict[str, Any]
    max_tokens: int = 2000
    temperature: float = 0.1

@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis"""
    success: bool
    analysis_type: str
    result_data: Dict[str, Any]
    confidence: float
    cost_estimate: float
    tokens_used: int
    processing_time: float
    suggestions: List[str]
    errors: List[str] = None

class AnalysisType(Enum):
    """Types of LLM analysis"""
    FIELD_DETECTION = "field_detection"
    CELL_RESOLUTION = "cell_resolution"
    MAPPING_VALIDATION = "mapping_validation"
    IMPROVEMENT_SUGGESTIONS = "improvement_suggestions"
    CONTEXT_ENHANCEMENT = "context_enhancement"

class EnhancedLLMIntegrationManager:
    """Enhanced LLM integration with grant-specific intelligence"""
    
    def __init__(self, llm_client, cost_guard):
        self.llm_client = llm_client
        self.cost_guard = cost_guard
        self.logger = logging.getLogger(__name__)
        self.grant_context_prompts = self._initialize_grant_prompts()
        
    def _initialize_grant_prompts(self) -> Dict[str, str]:
        """Initialize grant-specific prompts for different analysis types"""
        return {
            'field_detection': """
You are an expert in grant proposal budget analysis. Analyze the following template text and identify ALL placeholders/fields, paying special attention to:

1. PERSONNEL FIELDS: Names, titles, salaries, effort percentages, roles
2. EXPENSE FIELDS: Equipment items, costs, categories, descriptions  
3. DESCRIPTIVE FIELDS: Notes, justifications, descriptions, explanations
4. TEMPORAL FIELDS: Years, dates, project periods
5. QUANTITATIVE FIELDS: Quantities, rates, totals, percentages

For each field found, determine:
- Field type (personnel_name, personnel_title, personnel_salary, personnel_effort, expense_item, expense_amount, expense_category, description, notes, justification, year, date, quantity, rate, total)
- Confidence level (0.0-1.0)
- Context clues from surrounding text
- Specific suggestions for what to look for in budget data

IMPORTANT: Look for both explicit placeholders like {field_name} and implicit placeholders like [field] or __field__.

Template text to analyze:
{template_text}

Respond with a JSON object containing an array of detected fields.
""",

            'cell_resolution': """
You are an expert in matching grant budget template fields to spreadsheet data. Given the field information and budget cell data, identify the BEST matches by considering:

1. SEMANTIC MATCHING: Field names vs cell values and context
2. DATA TYPE COMPATIBILITY: Numeric fields need numeric cells, text fields need text cells
3. CONTEXTUAL RELEVANCE: Surrounding labels, column headers, row context
4. GRANT-SPECIFIC PATTERNS: Personnel sections, expense categories, notes columns
5. PROXIMITY LOGIC: Related data usually appears near each other

Field to match: {field_info}
Available budget cells: {cell_data}

For each potential match, provide:
- Match confidence (0.0-1.0)
- Specific reasons for the match
- Data type compatibility assessment
- Context relevance score

Focus especially on:
- Names and titles in personnel sections
- Dollar amounts in cost columns  
- Descriptions and notes (often in rightmost columns)
- Effort percentages and FTE values
- Equipment and supply lists

Respond with JSON containing ranked match suggestions.
""",

            'mapping_validation': """
You are validating field-to-cell mappings for a grant proposal budget. Review each mapping and assess:

1. LOGICAL CONSISTENCY: Does the mapping make sense?
2. DATA TYPE MATCH: Are numeric fields mapped to numbers, text to text?
3. CONTEXTUAL APPROPRIATENESS: Does the cell context match the field purpose?
4. COMPLETENESS: Are all important fields mapped?
5. ACCURACY: Are the mappings precise and specific?

Current mappings: {mappings}
Template context: {template_context}
Budget context: {budget_context}

For each mapping, provide:
- Validation confidence (0.0-1.0)
- Issues or concerns
- Improvement suggestions
- Alternative mapping recommendations

Pay special attention to:
- Personnel names mapping to actual names
- Salary fields mapping to dollar amounts
- Effort percentages mapping to percentage values
- Notes/descriptions mapping to text content
- Equipment items mapping to item names

Respond with JSON containing validation results and suggestions.
""",

            'improvement_suggestions': """
You are providing recommendations to improve grant budget analysis. Based on the analysis results, suggest:

1. TEMPLATE IMPROVEMENTS: Better field names, clearer placeholders
2. BUDGET ORGANIZATION: How to structure budget data better
3. MAPPING ENHANCEMENTS: Ways to improve field-to-cell matching
4. PROCESS OPTIMIZATION: Steps to streamline the workflow
5. QUALITY ASSURANCE: Methods to validate accuracy

Analysis results: {analysis_results}
Current issues: {issues}
User context: {user_context}

Provide specific, actionable suggestions including:
- Field naming conventions for better recognition
- Budget layout recommendations
- Common patterns to watch for
- Quality checks to perform
- Best practices for grant proposals

Respond with JSON containing categorized improvement suggestions.
""",

            'context_enhancement': """
You are enhancing the contextual understanding of budget data for grant proposal analysis. Analyze the budget structure and identify:

1. PERSONNEL SECTIONS: Groups of cells containing staff information
2. EXPENSE CATEGORIES: Equipment, travel, supplies, etc.
3. DESCRIPTIVE CONTENT: Notes, justifications, explanations
4. TEMPORAL ELEMENTS: Years, project phases, timelines
5. RELATIONSHIPS: How different data elements connect

Budget data summary: {budget_summary}
Detected patterns: {patterns}

Identify and describe:
- Personnel groupings and their components
- Expense category structures
- Notes and description locations
- Temporal organization patterns
- Cross-references and relationships

This will help improve automatic field detection and matching.

Respond with JSON containing enhanced context information.
"""
        }
    
    async def analyze_template_fields(self, template_text: str, 
                                    additional_context: Dict[str, Any] = None) -> LLMAnalysisResult:
        """Analyze template text to detect fields using LLM"""
        try:
            # Prepare prompt
            prompt = self.grant_context_prompts['field_detection'].format(
                template_text=template_text
            )
            
            # Add additional context if provided
            if additional_context:
                prompt += f"\n\nAdditional context: {json.dumps(additional_context, indent=2)}"
            
            # Make LLM request
            request = LLMAnalysisRequest(
                template_text=template_text,
                budget_summary={},
                analysis_type=AnalysisType.FIELD_DETECTION.value,
                context=additional_context or {},
                max_tokens=2000,
                temperature=0.1
            )
            
            result = await self._make_llm_request(request, prompt)
            
            # Parse and validate results
            if result.success:
                result.result_data = self._validate_field_detection_result(result.result_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in template field analysis: {str(e)}")
            return LLMAnalysisResult(
                success=False,
                analysis_type=AnalysisType.FIELD_DETECTION.value,
                result_data={},
                confidence=0.0,
                cost_estimate=0.0,
                tokens_used=0,
                processing_time=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def resolve_field_mappings(self, detected_fields: List[DetectedField], 
                                   budget_cells: List[CellData],
                                   batch_size: int = 5) -> LLMAnalysisResult:
        """Resolve field mappings using LLM in batches"""
        try:
            all_mappings = []
            total_cost = 0.0
            total_tokens = 0
            
            # Process fields in batches to manage cost and token limits
            for i in range(0, len(detected_fields), batch_size):
                batch_fields = detected_fields[i:i + batch_size]
                
                # Prepare field and cell data for LLM
                field_data = self._prepare_field_data_for_llm(batch_fields)
                cell_data = self._prepare_cell_data_for_llm(budget_cells)
                
                # Create prompt for this batch
                prompt = self.grant_context_prompts['cell_resolution'].format(
                    field_info=json.dumps(field_data, indent=2),
                    cell_data=json.dumps(cell_data, indent=2)
                )
                
                request = LLMAnalysisRequest(
                    template_text="",
                    budget_summary={'cells_count': len(budget_cells)},
                    analysis_type=AnalysisType.CELL_RESOLUTION.value,
                    context={'batch': i // batch_size + 1, 'total_batches': (len(detected_fields) + batch_size - 1) // batch_size},
                    max_tokens=3000,
                    temperature=0.1
                )
                
                batch_result = await self._make_llm_request(request, prompt)
                
                if batch_result.success:
                    batch_mappings = self._validate_cell_resolution_result(batch_result.result_data)
                    all_mappings.extend(batch_mappings)
                    total_cost += batch_result.cost_estimate
                    total_tokens += batch_result.tokens_used
                else:
                    self.logger.warning(f"Batch {i // batch_size + 1} failed: {batch_result.errors}")
            
            return LLMAnalysisResult(
                success=len(all_mappings) > 0,
                analysis_type=AnalysisType.CELL_RESOLUTION.value,
                result_data={'mappings': all_mappings},
                confidence=sum(m.get('confidence', 0) for m in all_mappings) / len(all_mappings) if all_mappings else 0,
                cost_estimate=total_cost,
                tokens_used=total_tokens,
                processing_time=0.0,  # Would need to track actual time
                suggestions=self._generate_mapping_suggestions(all_mappings)
            )
            
        except Exception as e:
            self.logger.error(f"Error in field mapping resolution: {str(e)}")
            return LLMAnalysisResult(
                success=False,
                analysis_type=AnalysisType.CELL_RESOLUTION.value,
                result_data={},
                confidence=0.0,
                cost_estimate=0.0,
                tokens_used=0,
                processing_time=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def validate_mappings(self, resolution_results: List[ResolutionResult],
                              template_context: str, budget_context: Dict[str, Any]) -> LLMAnalysisResult:
        """Validate field-to-cell mappings using LLM"""
        try:
            # Prepare mapping data for validation
            mappings_data = self._prepare_mappings_for_validation(resolution_results)
            
            prompt = self.grant_context_prompts['mapping_validation'].format(
                mappings=json.dumps(mappings_data, indent=2),
                template_context=template_context,
                budget_context=json.dumps(budget_context, indent=2)
            )
            
            request = LLMAnalysisRequest(
                template_text=template_context,
                budget_summary=budget_context,
                analysis_type=AnalysisType.MAPPING_VALIDATION.value,
                context={'mappings_count': len(resolution_results)},
                max_tokens=2500,
                temperature=0.1
            )
            
            result = await self._make_llm_request(request, prompt)
            
            if result.success:
                result.result_data = self._validate_mapping_validation_result(result.result_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in mapping validation: {str(e)}")
            return LLMAnalysisResult(
                success=False,
                analysis_type=AnalysisType.MAPPING_VALIDATION.value,
                result_data={},
                confidence=0.0,
                cost_estimate=0.0,
                tokens_used=0,
                processing_time=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def suggest_improvements(self, analysis_results: Dict[str, Any],
                                 issues: List[str], user_context: Dict[str, Any]) -> LLMAnalysisResult:
        """Generate improvement suggestions using LLM"""
        try:
            prompt = self.grant_context_prompts['improvement_suggestions'].format(
                analysis_results=json.dumps(analysis_results, indent=2),
                issues=json.dumps(issues),
                user_context=json.dumps(user_context, indent=2)
            )
            
            request = LLMAnalysisRequest(
                template_text="",
                budget_summary=user_context,
                analysis_type=AnalysisType.IMPROVEMENT_SUGGESTIONS.value,
                context={'issues_count': len(issues)},
                max_tokens=2000,
                temperature=0.2  # Slightly higher for creative suggestions
            )
            
            result = await self._make_llm_request(request, prompt)
            
            if result.success:
                result.result_data = self._validate_improvement_suggestions_result(result.result_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {str(e)}")
            return LLMAnalysisResult(
                success=False,
                analysis_type=AnalysisType.IMPROVEMENT_SUGGESTIONS.value,
                result_data={},
                confidence=0.0,
                cost_estimate=0.0,
                tokens_used=0,
                processing_time=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def enhance_context_understanding(self, budget_book: EnhancedBudgetBook) -> LLMAnalysisResult:
        """Enhance understanding of budget structure and context using LLM"""
        try:
            # Prepare budget summary for analysis
            budget_summary = self._prepare_budget_summary_for_llm(budget_book)
            patterns = self._extract_patterns_from_budget(budget_book)
            
            prompt = self.grant_context_prompts['context_enhancement'].format(
                budget_summary=json.dumps(budget_summary, indent=2),
                patterns=json.dumps(patterns, indent=2)
            )
            
            request = LLMAnalysisRequest(
                template_text="",
                budget_summary=budget_summary,
                analysis_type=AnalysisType.CONTEXT_ENHANCEMENT.value,
                context={'sheets_count': len(budget_book.sheets_data)},
                max_tokens=2500,
                temperature=0.1
            )
            
            result = await self._make_llm_request(request, prompt)
            
            if result.success:
                result.result_data = self._validate_context_enhancement_result(result.result_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in context enhancement: {str(e)}")
            return LLMAnalysisResult(
                success=False,
                analysis_type=AnalysisType.CONTEXT_ENHANCEMENT.value,  
                result_data={},
                confidence=0.0,
                cost_estimate=0.0,
                tokens_used=0,
                processing_time=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    async def _make_llm_request(self, request: LLMAnalysisRequest, prompt: str) -> LLMAnalysisResult:
        """Make LLM request with cost tracking and error handling"""
        try:
            import time
            start_time = time.time()
            
            # Check cost limits
            if not self.cost_guard.check_request_budget(estimated_tokens=request.max_tokens):
                raise Exception("Cost limit exceeded")
            
            # Make the actual LLM call
            response = await self.llm_client.complete(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            processing_time = time.time() - start_time
            
            # Parse JSON response
            try:
                result_data = json.loads(response.content)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    raise Exception("Could not parse JSON response from LLM")
            
            # Track costs
            cost_estimate = self.cost_guard.estimate_cost(response.usage.total_tokens)
            self.cost_guard.track_usage(response.usage.total_tokens, cost_estimate)
            
            return LLMAnalysisResult(
                success=True,
                analysis_type=request.analysis_type,
                result_data=result_data,
                confidence=0.8,  # Default confidence, can be adjusted based on response quality
                cost_estimate=cost_estimate,
                tokens_used=response.usage.total_tokens,
                processing_time=processing_time,
                suggestions=result_data.get('suggestions', [])
            )
            
        except Exception as e:
            self.logger.error(f"LLM request failed: {str(e)}")
            return LLMAnalysisResult(
                success=False,
                analysis_type=request.analysis_type,
                result_data={},
                confidence=0.0,
                cost_estimate=0.0,
                tokens_used=0,
                processing_time=0.0,
                suggestions=[],
                errors=[str(e)]
            )
    
    def _prepare_field_data_for_llm(self, fields: List[DetectedField]) -> List[Dict[str, Any]]:
        """Prepare field data for LLM analysis"""
        field_data = []
        for field in fields:
            field_dict = {
                'placeholder': field.placeholder,
                'original_text': field.original_text,
                'field_type': field.field_type.value,
                'confidence': field.confidence,
                'context_before': field.context_before[:100],  # Limit context length
                'context_after': field.context_after[:100],
                'grant_specific': field.grant_specific,
                'suggested_mappings': field.suggested_mappings[:3]  # Top 3 suggestions
            }
            field_data.append(field_dict)
        return field_data
    
    def _prepare_cell_data_for_llm(self, cells: List[CellData], max_cells: int = 50) -> List[Dict[str, Any]]:
        """Prepare cell data for LLM analysis (limit size to manage tokens)"""
        # Sort cells by confidence and take top cells
        sorted_cells = sorted(cells, key=lambda x: x.confidence, reverse=True)[:max_cells]
        
        cell_data = []
        for cell in sorted_cells:
            cell_dict = {
                'value': str(cell.value)[:100] if cell.value else "",  # Limit length
                'row': cell.row,
                'col': cell.col,
                'column_name': cell.column_name,
                'data_type': cell.data_type,
                'confidence': cell.confidence,
                'context_labels': cell.context_labels[:3],  # Top 3 context labels
                'notes': cell.notes[:100] if cell.notes else ""  # Limit notes length
            }
            cell_data.append(cell_dict)
        return cell_data
    
    def _prepare_mappings_for_validation(self, results: List[ResolutionResult]) -> List[Dict[str, Any]]:
        """Prepare mapping data for validation"""
        mappings_data = []
        for result in results:
            mapping_dict = {
                'field_placeholder': result.field.placeholder,
                'field_type': result.field.field_type.value,
                'field_confidence': result.field.confidence,
                'resolution_confidence': result.resolution_confidence,
                'needs_manual_review': result.needs_manual_review
            }
            
            if result.primary_match:
                mapping_dict['matched_cell'] = {
                    'value': str(result.primary_match.cell.value),
                    'data_type': result.primary_match.cell.data_type,
                    'context_labels': result.primary_match.cell.context_labels,
                    'match_reasons': result.primary_match.match_reasons
                }
            else:
                mapping_dict['matched_cell'] = None
            
            mappings_data.append(mapping_dict)
        return mappings_data
    
    def _prepare_budget_summary_for_llm(self, budget_book: EnhancedBudgetBook) -> Dict[str, Any]:
        """Prepare budget summary for LLM analysis"""
        summary = budget_book._generate_summary()
        
        # Add structure information
        summary['sheets'] = {}
        for sheet_name, sheet_data in budget_book.sheets_data.items():
            summary['sheets'][sheet_name] = {
                'max_row': sheet_data['max_row'],
                'max_col': sheet_data['max_col'],
                'notes_column': sheet_data.get('notes_column'),
                'personnel_entries_count': len(sheet_data.get('personnel_entries', [])),
                'expense_entries_count': len(sheet_data.get('expense_categories', [])),
                'contextual_groups_count': len(sheet_data.get('contextual_groups', []))
            }
        
        return summary
    
    def _extract_patterns_from_budget(self, budget_book: EnhancedBudgetBook) -> Dict[str, Any]:
        """Extract patterns from budget data"""
        patterns = {
            'common_labels': [],
            'numeric_patterns': [],
            'text_patterns': [],
            'structural_patterns': []
        }
        
        # Extract common context labels
        all_labels = []
        for cell in budget_book.all_cells:
            all_labels.extend(cell.context_labels)
        
        # Find most common labels
        from collections import Counter
        label_counts = Counter(all_labels)
        patterns['common_labels'] = [label for label, count in label_counts.most_common(10)]
        
        # Identify structural patterns
        for sheet_name, sheet_data in budget_book.sheets_data.items():
            if sheet_data.get('personnel_entries'):
                patterns['structural_patterns'].append(f"Personnel section identified in {sheet_name}")
            if sheet_data.get('expense_categories'):
                patterns['structural_patterns'].append(f"Expense categories found in {sheet_name}")
            if sheet_data.get('notes_column'):
                patterns['structural_patterns'].append(f"Notes column at position {sheet_data['notes_column']} in {sheet_name}")
        
        return patterns
    
    def _validate_field_detection_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean field detection results from LLM"""
        # Ensure required fields are present
        if 'fields' not in result_data:
            result_data['fields'] = []
        
        # Validate each detected field
        validated_fields = []
        for field_data in result_data.get('fields', []):
            if self._is_valid_field_data(field_data):
                validated_fields.append(field_data)
        
        result_data['fields'] = validated_fields
        return result_data
    
    def _validate_cell_resolution_result(self, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and clean cell resolution results from LLM"""
        mappings = result_data.get('mappings', [])
        validated_mappings = []
        
        for mapping in mappings:
            if self._is_valid_mapping_data(mapping):
                validated_mappings.append(mapping)
        
        return validated_mappings
    
    def _validate_mapping_validation_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mapping validation results from LLM"""
        if 'validations' not in result_data:
            result_data['validations'] = []
        
        return result_data
    
    def _validate_improvement_suggestions_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate improvement suggestions from LLM"""
        if 'suggestions' not in result_data:
            result_data['suggestions'] = []
        
        return result_data
    
    def _validate_context_enhancement_result(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate context enhancement results from LLM"""
        required_keys = ['personnel_sections', 'expense_categories', 'descriptive_content', 'relationships']
        for key in required_keys:
            if key not in result_data:
                result_data[key] = []
        
        return result_data
    
    def _is_valid_field_data(self, field_data: Dict[str, Any]) -> bool:
        """Check if field data is valid"""
        required_keys = ['placeholder', 'field_type', 'confidence']
        return all(key in field_data for key in required_keys)
    
    def _is_valid_mapping_data(self, mapping_data: Dict[str, Any]) -> bool:
        """Check if mapping data is valid"""
        required_keys = ['field_id', 'cell_matches', 'confidence']
        return all(key in mapping_data for key in required_keys)
    
    def _generate_mapping_suggestions(self, mappings: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on mapping results"""
        suggestions = []
        
        low_confidence_mappings = [m for m in mappings if m.get('confidence', 0) < 0.6]
        if low_confidence_mappings:
            suggestions.append(f"Review {len(low_confidence_mappings)} low-confidence mappings")
        
        unmatched_fields = [m for m in mappings if not m.get('cell_matches')]
        if unmatched_fields:
            suggestions.append(f"Manually map {len(unmatched_fields)} unmatched fields")
        
        return suggestions
    
    def get_analysis_summary(self, results: List[LLMAnalysisResult]) -> Dict[str, Any]:
        """Generate summary of all LLM analysis results"""
        total_cost = sum(r.cost_estimate for r in results)
        total_tokens = sum(r.tokens_used for r in results)
        successful_analyses = len([r for r in results if r.success])
        
        return {
            'total_analyses': len(results),
            'successful_analyses': successful_analyses,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'average_confidence': sum(r.confidence for r in results) / len(results) if results else 0,
            'analysis_types': [r.analysis_type for r in results]
        }