"""
Enhanced LLM integration manager with improved prompts for string analysis,
contextual field mapping, and grant-specific terminology recognition.
"""
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import re
from datetime import datetime

from cell_resolver import CellResolver, FieldMatch, CellContext
from field_detector import FieldDetector, TemplateField
from budget_book import BudgetBook, PersonnelEntry, BudgetSection

@dataclass
class LLMAnalysisRequest:
    """Request structure for LLM analysis."""
    analysis_type: str  # 'field_mapping', 'personnel_extraction', 'context_analysis'
    template_fields: List[TemplateField]
    budget_data: Dict[str, Any]
    context_hint: str = ""
    focus_areas: List[str] = None
    max_tokens: int = 1000

@dataclass
class LLMAnalysisResult:
    """Result structure from LLM analysis."""
    analysis_type: str
    results: Dict[str, Any]
    confidence: float
    reasoning: str
    suggestions: List[str] = None
    warnings: List[str] = None

class LLMIntegrationManager:
    """Enhanced LLM integration with focus on string analysis and contextual understanding."""
    
    def __init__(self, llm_client, cost_guard=None, enable_debug=False):
        self.llm_client = llm_client
        self.cost_guard = cost_guard
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize component managers
        self.cell_resolver = CellResolver(llm_client, enable_debug)
        self.field_detector = FieldDetector(enable_debug)
        self.budget_book = BudgetBook(enable_debug)
        
        # Analysis history for learning
        self.analysis_history = []
        
        # Grant-specific prompt templates
        self.prompt_templates = {
            'field_mapping_enhanced': """
You are an expert grant budget analyst. Your task is to map template fields to budget spreadsheet data with special attention to string values like names, titles, descriptions, and notes.

TEMPLATE FIELDS TO MAP:
{template_fields}

BUDGET SPREADSHEET CONTEXT:
- Total sheets: {sheet_count}
- Personnel found: {personnel_count}
- Budget sections: {section_types}

BUDGET DATA SAMPLE:
{budget_sample}

SPECIAL FOCUS AREAS:
1. PERSONNEL INFORMATION: Look for names, titles, positions, roles. These might be in text cells near salary amounts.
2. NOTES AND DESCRIPTIONS: Check rightmost columns for notes, justifications, descriptions. Look for column headers like "Notes", "Description", "Comments", "Justification".
3. GRANT-SPECIFIC TERMINOLOGY: Watch for terms like "Principal Investigator", "Co-PI", "Postdoc", "Key Personnel", "Equipment", "Travel", "Release Time".
4. CONTEXTUAL RELATIONSHIPS: A person's name might be in one cell with their title in an adjacent cell, and salary in a nearby numeric cell.
5. PROXIMITY MAPPING: The closest meaningful text label to a value is usually the correct mapping.

ANALYSIS INSTRUCTIONS:
- Pay equal attention to text strings and numeric values
- Look for patterns where labels appear near values (left, above, or as headers)
- Consider multi-cell entries (name in one cell, title in another)
- Don't ignore cells just because they're text - they often contain the most important information
- For grant budgets, personnel names and titles are as important as dollar amounts
- Notes columns often contain crucial justification text

For each template field, provide:
1. Best matching cell location(s) - may be multiple cells for complex fields
2. Confidence score (0-1)
3. Reasoning explaining why this mapping makes sense
4. Any contextual clues that support the mapping
5. Alternative candidates if confidence is not high

Response format: JSON with field mappings and detailed reasoning.
""",
            
            'personnel_extraction_enhanced': """
You are analyzing a grant budget spreadsheet to extract personnel information. Focus on finding ALL people mentioned, their roles, and associated details.

SPREADSHEET DATA:
{budget_data}

EXTRACTION GOALS:
1. NAMES: Find all person names, including variations like "Dr. Smith", "Smith, John", "J. Smith"
2. TITLES/ROLES: Look for positions like "Principal Investigator", "Co-PI", "Postdoc", "Graduate Student", "Staff Scientist"
3. EFFORT/TIME: Find percentage effort, FTE, person-months, or time commitments
4. COSTS: Identify salary, fringe, or total costs associated with each person
5. NOTES: Capture any additional information, justifications, or descriptions about personnel

ANALYSIS APPROACH:
- Scan entire spreadsheet, not just obvious "Personnel" sections
- Names might appear in various formats and locations
- Titles could be abbreviations (PI, Co-PI, RA, etc.)
- Look for relationships between adjacent cells (name next to title next to salary)
- Check for notes or descriptions that provide context about roles
- Consider that one person might appear multiple times (different years, different roles)

OUTPUT FORMAT:
For each person found, provide:
- Name (as found in spreadsheet)
- Title/Role (including variations)
- Effort percentage or FTE if found
- Associated costs if identifiable
- Location in spreadsheet (sheet, row, column)
- Confidence in extraction (0-1)
- Any additional notes or context

Return as structured JSON.
""",
            
            'context_analysis_enhanced': """
You are analyzing the context and structure of a grant budget spreadsheet to improve field mapping accuracy.

SPREADSHEET OVERVIEW:
{spreadsheet_overview}

ANALYSIS OBJECTIVES:
1. STRUCTURAL ANALYSIS: Identify how the spreadsheet is organized (sections, tables, layouts)
2. CONTENT PATTERNS: Find patterns in how information is presented
3. TEXT vs NUMERIC DISTRIBUTION: Understand where text information appears vs numeric data
4. RELATIONSHIP MAPPING: Identify how labels relate to values (proximity, headers, etc.)
5. GRANT-SPECIFIC PATTERNS: Look for typical grant budget structures and terminology

SPECIFIC FOCUS:
- How are personnel organized? (table format, list format, mixed)
- Where do descriptions and notes typically appear? (rightmost columns, separate sections)
- What labeling conventions are used? (headers, left-side labels, inline labels)
- Are there multi-year or multi-category breakdowns?
- How detailed are the text descriptions?

CONTEXT CLUES TO IDENTIFY:
- Column headers that indicate content type
- Section headers or dividers
- Patterns in cell formatting or positioning
- Relationships between text and numeric cells
- Common grant budget categories (Personnel, Equipment, Travel, etc.)

PROVIDE:
1. Structural summary of the spreadsheet
2. Content distribution analysis (where text vs numbers appear)
3. Labeling and organization patterns
4. Recommendations for improving field mapping
5. Specific areas where manual review might be needed

Return analysis as structured JSON with clear sections.
""",
            
            'field_matching_enhanced': """
You are matching specific template fields to budget spreadsheet values with enhanced contextual analysis.

FIELD TO MATCH: {field_name}
FIELD TYPE: {field_type}
FIELD CONTEXT: {field_context}

CANDIDATE VALUES FROM SPREADSHEET:
{candidate_values}

MATCHING CRITERIA:
1. CONTENT RELEVANCE: Does the spreadsheet value match what the field is asking for?
2. TYPE COMPATIBILITY: For text fields, prioritize string values; for numeric fields, prioritize numbers
3. CONTEXTUAL CLUES: Consider nearby labels, column headers, section context
4. GRANT DOMAIN KNOWLEDGE: Apply knowledge of typical grant budget structures
5. SEMANTIC SIMILARITY: Match meaning even if exact words differ

SPECIAL CONSIDERATIONS:
- Personnel fields should match to names, titles, or roles
- Cost fields should match to monetary values or their descriptions
- Description fields should match to longer text explanations
- Notes fields often appear in rightmost columns
- Consider compound matches (multiple cells for one field)

For the field "{field_name}":
1. Rank all candidates by relevance (1-10 scale)
2. Explain reasoning for top matches
3. Identify if multiple cells might together fulfill the field requirement
4. Note any contextual information that supports or contradicts matches
5. Suggest if the field might not have a match in this dataset

Respond with ranked matches and detailed explanations in JSON format.
"""
        }
    
    def analyze_budget_with_enhanced_llm(self, template_path: str, budget_path: str) -> Dict[str, Any]:
        """Comprehensive budget analysis with enhanced LLM integration."""
        self.logger.info("Starting enhanced LLM budget analysis")
        
        # Step 1: Load and analyze template
        template_fields = self.field_detector.analyze_template(template_path)
        self.logger.info(f"Detected {len(template_fields)} template fields")
        
        # Step 2: Load and parse budget
        budget_data = self.budget_book.load_budget(budget_path)
        self.logger.info(f"Loaded budget with {len(budget_data['sheets'])} sheets")
        
        # Step 3: Comprehensive analysis
        analysis_results = {
            'template_analysis': {
                'fields': [asdict(field) for field in template_fields],
                'summary': self.field_detector.get_field_summary(template_fields)
            },
            'budget_analysis': self._analyze_budget_structure(budget_data),
            'personnel_analysis': self._extract_personnel_with_llm(budget_data),
            'field_mappings': self._perform_enhanced_field_mapping(template_fields, budget_data),
            'quality_assessment': self._assess_mapping_quality(template_fields, budget_data),
            'recommendations': self._generate_recommendations(template_fields, budget_data)
        }
        
        return analysis_results
    
    def _analyze_budget_structure(self, budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze budget structure with LLM assistance."""
        if not self.llm_client:
            return {'error': 'LLM client not available'}
        
        # Prepare budget overview for LLM
        overview = self._prepare_budget_overview(budget_data)
        
        # Request structural analysis from LLM
        request = LLMAnalysisRequest(
            analysis_type='context_analysis',
            template_fields=[],
            budget_data=budget_data,
            context_hint="Focus on spreadsheet structure and organization patterns"
        )
        
        try:
            prompt = self.prompt_templates['context_analysis_enhanced'].format(
                spreadsheet_overview=json.dumps(overview, indent=2)
            )
            
            response = self.llm_client.analyze_context(prompt, max_tokens=800)
            result = self._parse_llm_response(response, 'context_analysis')
            
            return {
                'llm_analysis': result,
                'basic_stats': overview,
                'confidence': result.confidence if result else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"LLM structure analysis failed: {e}")
            return {
                'error': str(e),
                'basic_stats': overview,
                'confidence': 0.3
            }
    
    def _extract_personnel_with_llm(self, budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract personnel information using enhanced LLM analysis."""
        if not self.llm_client:
            # Fallback to rule-based extraction
            all_personnel = []
            for sheet_name, sheet_data in budget_data['sheets'].items():
                personnel = sheet_data.get('personnel', [])
                all_personnel.extend([asdict(p) for p in personnel])
            return {'personnel': all_personnel, 'method': 'rule_based'}
        
        try:
            # Prepare data sample for LLM
            budget_sample = self._prepare_budget_sample_for_personnel(budget_data)
            
            prompt = self.prompt_templates['personnel_extraction_enhanced'].format(
                budget_data=json.dumps(budget_sample, indent=2)
            )
            
            response = self.llm_client.analyze_context(prompt, max_tokens=1200)
            result = self._parse_llm_response(response, 'personnel_extraction')
            
            return {
                'llm_personnel': result.results if result else {},
                'rule_based_personnel': self._get_rule_based_personnel(budget_data),
                'method': 'llm_enhanced',
                'confidence': result.confidence if result else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"LLM personnel extraction failed: {e}")
            return {
                'error': str(e),
                'rule_based_personnel': self._get_rule_based_personnel(budget_data),
                'method': 'fallback'
            }
    
    def _perform_enhanced_field_mapping(self, template_fields: List[TemplateField], 
                                      budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform enhanced field mapping with LLM assistance."""
        if not self.llm_client:
            # Fallback to rule-based mapping
            return self._perform_rule_based_mapping(template_fields, budget_data)
        
        try:
            # Prepare comprehensive context
            mapping_context = self._prepare_mapping_context(template_fields, budget_data)
            
            prompt = self.prompt_templates['field_mapping_enhanced'].format(
                template_fields=self._format_template_fields(template_fields),
                sheet_count=len(budget_data['sheets']),
                personnel_count=self._count_personnel(budget_data),
                section_types=self._get_section_types(budget_data),
                budget_sample=json.dumps(mapping_context, indent=2)
            )
            
            response = self.llm_client.analyze_context(prompt, max_tokens=1500)
            result = self._parse_llm_response(response, 'field_mapping')
            
            # Combine LLM results with rule-based fallbacks
            combined_results = self._combine_mapping_results(
                result.results if result else {},
                self._perform_rule_based_mapping(template_fields, budget_data)
            )
            
            return {
                'llm_mappings': result.results if result else {},
                'rule_based_mappings': self._perform_rule_based_mapping(template_fields, budget_data),
                'combined_mappings': combined_results,
                'method': 'llm_enhanced',
                'confidence': result.confidence if result else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced field mapping failed: {e}")
            return {
                'error': str(e),
                'rule_based_mappings': self._perform_rule_based_mapping(template_fields, budget_data),
                'method': 'fallback'
            }
    
    def _perform_rule_based_mapping(self, template_fields: List[TemplateField], 
                                  budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform rule-based field mapping as fallback."""
        mappings = {}
        
        for field in template_fields:
            # Use the enhanced cell resolver
            field_matches = []
            
            for sheet_name, sheet_data in budget_data['sheets'].items():
                df = sheet_data['dataframe']
                try:
                    matches = self.cell_resolver.resolve_field_mappings(df, [field.name])
                    for match in matches:
                        match_data = {
                            'sheet': sheet_name,
                            'cell_address': match.cell_address,
                            'value': match.value,
                            'confidence': match.confidence,
                            'match_type': match.match_type,
                            'notes': match.notes
                        }
                        field_matches.append(match_data)
                except Exception as e:
                    self.logger.warning(f"Rule-based mapping failed for {field.name}: {e}")
            
            # Sort by confidence and take top matches
            field_matches.sort(key=lambda x: x['confidence'], reverse=True)
            mappings[field.name] = field_matches[:3]  # Top 3 matches
        
        return mappings
    
    def _assess_mapping_quality(self, template_fields: List[TemplateField], 
                              budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of field mappings."""
        assessment = {
            'total_fields': len(template_fields),
            'mapped_fields': 0,
            'high_confidence_mappings': 0,
            'medium_confidence_mappings': 0,
            'low_confidence_mappings': 0,
            'unmapped_fields': [],
            'mapping_issues': [],
            'overall_quality': 'unknown'
        }
        
        # This would be populated based on the actual mapping results
        # For now, return the structure
        return assessment
    
    def _generate_recommendations(self, template_fields: List[TemplateField], 
                                budget_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving field mapping."""
        recommendations = []
        
        # Check for common issues
        text_fields = [f for f in template_fields if f.field_type in ['text', 'personnel', 'description']]
        if text_fields:
            recommendations.append(
                "Ensure the mapping tool examines text cells, not just numeric values. "
                "Personnel names, titles, and descriptions are often in text cells."
            )
        
        # Check for notes fields
        notes_fields = [f for f in template_fields if 'note' in f.name.lower() or 'description' in f.name.lower()]
        if notes_fields:
            recommendations.append(
                "Look for notes and descriptions in the rightmost columns of the spreadsheet. "
                "These columns may be labeled 'Notes', 'Description', 'Comments', or similar."
            )
        
        # Check for personnel fields
        personnel_fields = [f for f in template_fields if f.field_type == 'personnel']
        if personnel_fields:
            recommendations.append(
                "For personnel information, check multiple nearby cells. A person's name, title, "
                "and salary might be in adjacent cells rather than a single cell."
            )
        
        recommendations.append(
            "If automatic mapping has low confidence, consider using the manual override "
            "feature to specify correct mappings."
        )
        
        return recommendations
    
    def _prepare_budget_overview(self, budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a high-level overview of budget structure."""
        overview = {
            'total_sheets': len(budget_data['sheets']),
            'sheets': {}
        }
        
        for sheet_name, sheet_data in budget_data['sheets'].items():
            df = sheet_data['dataframe']
            cells = sheet_data.get('cells', [])
            
            # Count cell types
            text_cells = len([c for c in cells if c.data_type == 'text'])
            numeric_cells = len([c for c in cells if c.data_type in ['numeric', 'currency']])
            
            overview['sheets'][sheet_name] = {
                'shape': sheet_data.get('shape', (0, 0)),
                'total_cells': len(cells),
                'text_cells': text_cells,
                'numeric_cells': numeric_cells,
                'personnel_count': len(sheet_data.get('personnel', [])),
                'sections': [s.section_type for s in sheet_data.get('sections', [])]
            }
        
        return overview
    
    def _prepare_budget_sample_for_personnel(self, budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a focused sample of budget data for personnel extraction."""
        sample = {}
        
        for sheet_name, sheet_data in budget_data['sheets'].items():
            cells = sheet_data.get('cells', [])
            
            # Get text cells and nearby numeric cells
            relevant_cells = []
            for cell in cells[:100]:  # Limit sample size
                if (cell.data_type == 'text' or 
                    (cell.data_type in ['numeric', 'currency'] and 
                     any(tc.data_type == 'text' and 
                         abs(tc.row - cell.row) <= 1 and abs(tc.col - cell.col) <= 2 
                         for tc in cells))):
                    relevant_cells.append({
                        'row': cell.row,
                        'col': cell.col,
                        'value': cell.value,
                        'type': cell.data_type
                    })
            
            sample[sheet_name] = relevant_cells
        
        return sample
    
    def _prepare_mapping_context(self, template_fields: List[TemplateField], 
                               budget_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare comprehensive context for field mapping."""
        context = {
            'template_summary': {
                'total_fields': len(template_fields),
                'field_types': {},
                'priority_fields': []
            },
            'budget_summary': {},
            'sample_data': {}
        }
        
        # Summarize template fields
        for field in template_fields:
            field_type = field.field_type
            context['template_summary']['field_types'][field_type] = \
                context['template_summary']['field_types'].get(field_type, 0) + 1
            
            if field.confidence > 0.8:
                context['template_summary']['priority_fields'].append(field.name)
        
        # Sample budget data
        for sheet_name, sheet_data in budget_data['sheets'].items():
            cells = sheet_data.get('cells', [])
            
            # Get a diverse sample of cells
            sample_cells = []
            for i, cell in enumerate(cells):
                if i % 10 == 0 or cell.data_type == 'text':  # Every 10th cell + all text cells
                    sample_cells.append({
                        'location': f"R{cell.row}C{cell.col}",
                        'value': str(cell.value)[:100],  # Truncate long values
                        'type': cell.data_type
                    })
                
                if len(sample_cells) >= 50:  # Limit sample size
                    break
            
            context['sample_data'][sheet_name] = sample_cells
        
        return context
    
    def _format_template_fields(self, template_fields: List[TemplateField]) -> str:
        """Format template fields for LLM prompt."""
        formatted_fields = []
        
        for field in template_fields:
            field_info = f"- {field.name} (Type: {field.field_type})"
            if field.context:
                field_info += f"\n  Context: {field.context[:100]}..."
            formatted_fields.append(field_info)
        
        return "\n".join(formatted_fields)
    
    def _count_personnel(self, budget_data: Dict[str, Any]) -> int:
        """Count total personnel entries across all sheets."""
        total = 0
        for sheet_data in budget_data['sheets'].values():
            total += len(sheet_data.get('personnel', []))
        return total
    
    def _get_section_types(self, budget_data: Dict[str, Any]) -> List[str]:
        """Get all section types found in the budget."""
        section_types = set()
        for sheet_data in budget_data['sheets'].values():
            sections = sheet_data.get('sections', [])
            for section in sections:
                section_types.add(section.section_type)
        return list(section_types)
    
    def _get_rule_based_personnel(self, budget_data: Dict[str, Any]) -> List[Dict]:
        """Get personnel using rule-based extraction."""
        all_personnel = []
        for sheet_name, sheet_data in budget_data['sheets'].items():
            personnel = sheet_data.get('personnel', [])
            for person in personnel:
                person_dict = asdict(person)
                person_dict['sheet'] = sheet_name
                all_personnel.append(person_dict)
        return all_personnel
    
    def _combine_mapping_results(self, llm_results: Dict, rule_results: Dict) -> Dict:
        """Combine LLM and rule-based mapping results."""
        combined = {}
        
        # Start with rule-based results
        for field_name, mappings in rule_results.items():
            combined[field_name] = {
                'rule_based': mappings,
                'llm_based': llm_results.get(field_name, []),
                'final_recommendation': mappings[0] if mappings else None
            }
        
        # Add any LLM-only results
        for field_name, mappings in llm_results.items():
            if field_name not in combined:
                combined[field_name] = {
                    'rule_based': [],
                    'llm_based': mappings,
                    'final_recommendation': mappings[0] if mappings else None
                }
        
        return combined
    
    def _parse_llm_response(self, response: str, analysis_type: str) -> Optional[LLMAnalysisResult]:
        """Parse LLM response into structured result."""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                results = json.loads(response)
            else:
                # Parse as structured text
                results = self._parse_text_response(response, analysis_type)
            
            # Extract confidence if present
            confidence = results.get('confidence', 0.7)
            reasoning = results.get('reasoning', 'LLM analysis completed')
            
            return LLMAnalysisResult(
                analysis_type=analysis_type,
                results=results,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            self.logger.debug(f"Raw response: {response}")
            return None
    
    def _parse_text_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Parse text-based LLM response."""
        # Basic text parsing - this would need to be more sophisticated
        # based on the actual response format from your LLM
        
        lines = response.split('\n')
        results = {
            'raw_response': response,
            'confidence': 0.5,
            'reasoning': 'Text-based response parsing'
        }
        
        if analysis_type == 'field_mapping':
            results['mappings'] = {}
            # Parse field mappings from text
            # This would need specific parsing logic based on your LLM's output format
        
        elif analysis_type == 'personnel_extraction':
            results['personnel'] = []
            # Parse personnel entries from text
        
        elif analysis_type == 'context_analysis':
            results['structure_analysis'] = response
        
        return results

    def get_mapping_suggestions_for_field(self, field: TemplateField, 
                                        budget_data: Dict[str, Any]) -> List[Tuple[str, float, str]]:
        """Get specific mapping suggestions for a single field."""
        if not self.llm_client:
            return []
        
        try:
            # Prepare candidate values from budget
            candidates = self._extract_field_candidates(field, budget_data)
            
            prompt = self.prompt_templates['field_matching_enhanced'].format(
                field_name=field.name,
                field_type=field.field_type,
                field_context=field.context,
                candidate_values=json.dumps(candidates, indent=2)
            )
            
            response = self.llm_client.analyze_context(prompt, max_tokens=600)
            result = self._parse_llm_response(response, 'field_matching')
            
            if result and 'matches' in result.results:
                return [(m['location'], m['confidence'], m['reasoning']) 
                       for m in result.results['matches']]
            
        except Exception as e:
            self.logger.error(f"Field-specific mapping failed: {e}")
        
        return []
    
    def _extract_field_candidates(self, field: TemplateField, budget_data: Dict[str, Any]) -> List[Dict]:
        """Extract candidate values for a specific field."""
        candidates = []
        
        for sheet_name, sheet_data in budget_data['sheets'].items():
            cells = sheet_data.get('cells', [])
            
            for cell in cells:
                # Basic filtering based on field type
                is_candidate = False
                
                if field.field_type == 'personnel' and cell.data_type == 'text':
                    if self._might_be_person_name(str(cell.value)):
                        is_candidate = True
                
                elif field.field_type == 'currency' and cell.data_type in ['currency', 'numeric']:
                    is_candidate = True
                
                elif field.field_type == 'description' and cell.data_type == 'text':
                    if len(str(cell.value)) > 20:  # Longer text for descriptions
                        is_candidate = True
                
                elif field.field_type == 'text' and cell.data_type == 'text':
                    is_candidate = True
                
                if is_candidate:
                    candidates.append({
                        'location': f"{sheet_name}!R{cell.row}C{cell.col}",
                        'value': str(cell.value)[:200],  # Truncate long values
                        'type': cell.data_type,
                        'sheet': sheet_name
                    })
        
        return candidates[:50]  # Limit to top 50 candidates
    
    def _might_be_person_name(self, value: str) -> bool:
        """Quick check if a value might be a person's name."""
        if not isinstance(value, str) or len(value.strip()) < 3:
            return False
        
        words = value.strip().split()
        if len(words) < 2 or len(words) > 4:
            return False
        
        # Basic checks
        has_capitals = any(word[0].isupper() for word in words if word)
        no_numbers = not any(char.isdigit() for char in value)
        
        return has_capitals and no_numbers