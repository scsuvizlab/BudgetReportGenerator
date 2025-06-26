"""
Enhanced Cell Resolver - Intelligent mapping between template fields and budget cells
Uses context awareness, proximity matching, and grant-specific intelligence
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum
import difflib
from collections import defaultdict

# Import the enhanced structures
from enhanced_budget_book import CellData, EnhancedBudgetBook
from enhanced_field_detector import DetectedField, FieldType

@dataclass
class CellMatch:
    """A potential match between a template field and a budget cell"""
    cell: CellData
    field: DetectedField
    confidence: float
    match_reasons: List[str]
    proximity_score: float
    context_score: float
    semantic_score: float
    final_score: float

@dataclass
class ResolutionResult:
    """Result of cell resolution process"""
    field: DetectedField
    primary_match: Optional[CellMatch]
    alternative_matches: List[CellMatch]
    resolution_confidence: float
    needs_manual_review: bool
    suggestions: List[str]

class MatchType(Enum):
    """Types of matches found"""
    EXACT_MATCH = "exact"
    SEMANTIC_MATCH = "semantic"
    PROXIMITY_MATCH = "proximity"
    CONTEXT_MATCH = "context"
    FUZZY_MATCH = "fuzzy"
    HEURISTIC_MATCH = "heuristic"

class EnhancedCellResolver:
    """Enhanced cell resolver with intelligent matching capabilities"""
    
    def __init__(self, budget_book: EnhancedBudgetBook):
        self.budget_book = budget_book
        self.logger = logging.getLogger(__name__)
        self.semantic_patterns = self._initialize_semantic_patterns()
        self.proximity_weights = self._initialize_proximity_weights()
        
    def _initialize_semantic_patterns(self) -> Dict[FieldType, Dict[str, List[str]]]:
        """Initialize semantic patterns for different field types"""
        return {
            FieldType.PERSONNEL_NAME: {
                'exact': ['name', 'investigator', 'researcher', 'pi', 'person'],
                'partial': ['staff', 'faculty', 'team', 'personnel', 'employee'],
                'indicators': ['dr', 'prof', 'mr', 'ms', 'mrs', 'phd', 'md']
            },
            
            FieldType.PERSONNEL_TITLE: {
                'exact': ['title', 'position', 'role', 'rank'],
                'partial': ['professor', 'scientist', 'analyst', 'specialist', 'director'],
                'indicators': ['investigator', 'researcher', 'manager', 'coordinator']
            },
            
            FieldType.PERSONNEL_SALARY: {
                'exact': ['salary', 'wage', 'pay', 'compensation', 'cost'],
                'partial': ['annual', 'monthly', 'hourly', 'rate', 'amount'],
                'indicators': ['$', 'dollar', 'usd', 'total', 'sum']
            },
            
            FieldType.PERSONNEL_EFFORT: {
                'exact': ['effort', 'percent', 'percentage', 'fte', 'time'],
                'partial': ['commitment', 'allocation', 'dedication'],
                'indicators': ['%', 'pct', '0.', 'full', 'part']
            },
            
            FieldType.EXPENSE_ITEM: {
                'exact': ['item', 'equipment', 'supply', 'material', 'tool'],
                'partial': ['instrument', 'device', 'software', 'license', 'product'],
                'indicators': ['model', 'brand', 'type', 'version', 'unit']
            },
            
            FieldType.EXPENSE_AMOUNT: {
                'exact': ['cost', 'price', 'amount', 'total', 'expense'],
                'partial': ['budget', 'value', 'sum', 'fee', 'charge'],
                'indicators': ['$', 'dollar', 'usd', 'each', 'per']
            },
            
            FieldType.DESCRIPTION: {
                'exact': ['description', 'details', 'summary', 'explanation'],
                'partial': ['info', 'information', 'specification', 'overview'],
                'indicators': ['brief', 'detailed', 'full', 'complete']
            },
            
            FieldType.NOTES: {
                'exact': ['notes', 'note', 'comments', 'remarks', 'memo'],
                'partial': ['additional', 'extra', 'supplemental', 'other'],
                'indicators': ['see', 'refer', 'include', 'also']
            },
            
            FieldType.JUSTIFICATION: {
                'exact': ['justification', 'rationale', 'reason', 'why'],
                'partial': ['explanation', 'necessity', 'importance', 'purpose'],
                'indicators': ['because', 'needed', 'required', 'essential']
            }
        }
    
    def _initialize_proximity_weights(self) -> Dict[str, float]:
        """Initialize weights for proximity-based matching"""
        return {
            'same_row': 0.8,
            'adjacent_row': 0.6,
            'same_column': 0.4,
            'adjacent_column': 0.3,
            'diagonal': 0.2,
            'nearby': 0.1
        }
    
    def resolve_fields(self, detected_fields: List[DetectedField]) -> List[ResolutionResult]:
        """Resolve all detected fields to budget cells"""
        resolution_results = []
        
        # Get all budget cells for analysis
        all_cells = self.budget_book.all_cells
        
        for field in detected_fields:
            self.logger.info(f"Resolving field: {field.placeholder}")
            
            # Find potential matches
            potential_matches = self._find_potential_matches(field, all_cells)
            
            # Score and rank matches
            scored_matches = self._score_matches(field, potential_matches)
            
            # Determine best match and alternatives
            primary_match = scored_matches[0] if scored_matches else None
            alternative_matches = scored_matches[1:5]  # Top 4 alternatives
            
            # Calculate overall confidence
            resolution_confidence = primary_match.final_score if primary_match else 0.0
            
            # Determine if manual review is needed
            needs_manual_review = (
                not primary_match or 
                resolution_confidence < 0.6 or
                (len(scored_matches) > 1 and scored_matches[1].final_score > 0.7)
            )
            
            # Generate suggestions
            suggestions = self._generate_resolution_suggestions(field, scored_matches)
            
            result = ResolutionResult(
                field=field,
                primary_match=primary_match,
                alternative_matches=alternative_matches,
                resolution_confidence=resolution_confidence,
                needs_manual_review=needs_manual_review,
                suggestions=suggestions
            )
            
            resolution_results.append(result)
        
        self.logger.info(f"Resolved {len(resolution_results)} fields")
        return resolution_results
    
    def _find_potential_matches(self, field: DetectedField, all_cells: List[CellData]) -> List[CellData]:
        """Find potential cell matches for a field"""
        potential_matches = []
        
        # Get semantic patterns for this field type
        patterns = self.semantic_patterns.get(field.field_type, {})
        
        for cell in all_cells:
            if self._is_potential_match(field, cell, patterns):
                potential_matches.append(cell)
        
        return potential_matches
    
    def _is_potential_match(self, field: DetectedField, cell: CellData, 
                           patterns: Dict[str, List[str]]) -> bool:
        """Check if a cell could potentially match a field"""
        # Skip empty cells
        if not cell.value:
            return False
        
        cell_text = str(cell.value).lower()
        context_text = ' '.join(cell.context_labels).lower()
        column_text = cell.column_name.lower()
        
        # Check for exact field name match
        field_name = field.original_text.lower()
        if field_name in cell_text or field_name in context_text or field_name in column_text:
            return True
        
        # Check semantic patterns
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if (pattern in cell_text or pattern in context_text or 
                    pattern in column_text):
                    return True
        
        # Type-specific matching
        if field.field_type in [FieldType.PERSONNEL_SALARY, FieldType.EXPENSE_AMOUNT, 
                               FieldType.PERSONNEL_EFFORT, FieldType.QUANTITY]:
            # For numeric fields, check if cell contains numbers
            if cell.data_type in ['numeric', 'mixed']:
                return True
        
        elif field.field_type in [FieldType.PERSONNEL_NAME, FieldType.PERSONNEL_TITLE,
                                 FieldType.EXPENSE_ITEM, FieldType.DESCRIPTION, FieldType.NOTES]:
            # For text fields, check if cell contains substantial text
            if cell.data_type in ['text', 'mixed'] and len(str(cell.value)) > 2:
                return True
        
        # Check if cell is in notes column (for description/notes fields)
        if (field.field_type in [FieldType.DESCRIPTION, FieldType.NOTES, FieldType.JUSTIFICATION] and
            hasattr(self.budget_book, 'sheets_data')):
            for sheet_data in self.budget_book.sheets_data.values():
                if sheet_data.get('notes_column') == cell.col:
                    return True
        
        return False
    
    def _score_matches(self, field: DetectedField, potential_matches: List[CellData]) -> List[CellMatch]:
        """Score and rank potential matches"""
        scored_matches = []
        
        for cell in potential_matches:
            # Calculate component scores
            semantic_score = self._calculate_semantic_score(field, cell)
            context_score = self._calculate_context_score(field, cell)
            proximity_score = self._calculate_proximity_score(field, cell)
            
            # Calculate final score (weighted combination)
            final_score = (
                semantic_score * 0.4 +
                context_score * 0.35 +
                proximity_score * 0.25
            )
            
            # Determine match reasons
            match_reasons = self._determine_match_reasons(field, cell, semantic_score, 
                                                        context_score, proximity_score)
            
            # Create match object
            match = CellMatch(
                cell=cell,
                field=field,
                confidence=cell.confidence,
                match_reasons=match_reasons,
                proximity_score=proximity_score,
                context_score=context_score,
                semantic_score=semantic_score,
                final_score=final_score
            )
            
            scored_matches.append(match)
        
        # Sort by final score (descending)
        scored_matches.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_matches
    
    def _calculate_semantic_score(self, field: DetectedField, cell: CellData) -> float:
        """Calculate semantic similarity score"""
        score = 0.0
        
        field_name = field.original_text.lower()
        cell_text = str(cell.value).lower()
        
        # Exact match gets high score
        if field_name == cell_text:
            score += 0.9
        
        # Fuzzy string matching
        similarity = difflib.SequenceMatcher(None, field_name, cell_text).ratio()
        score += similarity * 0.6
        
        # Check semantic patterns
        patterns = self.semantic_patterns.get(field.field_type, {})
        
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern in cell_text:
                    if pattern_type == 'exact':
                        score += 0.4
                    elif pattern_type == 'partial':
                        score += 0.3
                    elif pattern_type == 'indicators':
                        score += 0.2
        
        # Data type compatibility
        if self._check_data_type_compatibility(field.field_type, cell.data_type):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_context_score(self, field: DetectedField, cell: CellData) -> float:
        """Calculate context-based score"""
        score = 0.0
        
        field_name = field.original_text.lower()
        context_text = ' '.join(cell.context_labels).lower()
        column_text = cell.column_name.lower()
        
        # Check if field name appears in context
        if field_name in context_text:
            score += 0.5
        if field_name in column_text:
            score += 0.4
        
        # Check for related terms in context
        patterns = self.semantic_patterns.get(field.field_type, {})
        for pattern_list in patterns.values():
            for pattern in pattern_list:
                if pattern in context_text:
                    score += 0.2
                if pattern in column_text:
                    score += 0.3
        
        # Grant-specific context boost
        if field.grant_specific:
            grant_terms = ['grant', 'proposal', 'budget', 'personnel', 'investigator']
            for term in grant_terms:
                if term in context_text:
                    score += 0.1
        
        # Cell confidence boost
        score += cell.confidence * 0.3
        
        return min(score, 1.0)
    
    def _calculate_proximity_score(self, field: DetectedField, cell: CellData) -> float:
        """Calculate proximity-based score (heuristic)"""
        # Since we don't have field positions in the budget, use contextual proximity
        score = 0.0
        
        # If cell has good context labels, boost proximity score
        if cell.context_labels:
            score += 0.3
        
        # If cell is in a structured group (like personnel or expense sections)
        if hasattr(self.budget_book, 'sheets_data'):
            for sheet_data in self.budget_book.sheets_data.values():
                # Check if cell is in personnel entries
                if any(entry['cell'] == cell for entry in sheet_data.get('personnel_entries', [])):
                    if field.field_type.value.startswith('personnel'):
                        score += 0.4
                
                # Check if cell is in expense entries
                if any(entry['cell'] == cell for entry in sheet_data.get('expense_categories', [])):
                    if field.field_type.value.startswith('expense'):
                        score += 0.4
                
                # Check if cell is in notes column
                if (sheet_data.get('notes_column') == cell.col and 
                    field.field_type in [FieldType.NOTES, FieldType.DESCRIPTION, FieldType.JUSTIFICATION]):
                    score += 0.5
        
        # Proximity based on data type appropriateness
        if self._check_data_type_compatibility(field.field_type, cell.data_type):
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_data_type_compatibility(self, field_type: FieldType, cell_data_type: str) -> bool:
        """Check if field type is compatible with cell data type"""
        numeric_fields = {
            FieldType.PERSONNEL_SALARY, FieldType.PERSONNEL_EFFORT, 
            FieldType.EXPENSE_AMOUNT, FieldType.QUANTITY, FieldType.RATE, 
            FieldType.TOTAL, FieldType.YEAR
        }
        
        text_fields = {
            FieldType.PERSONNEL_NAME, FieldType.PERSONNEL_TITLE,
            FieldType.EXPENSE_ITEM, FieldType.EXPENSE_CATEGORY,
            FieldType.DESCRIPTION, FieldType.NOTES, FieldType.JUSTIFICATION
        }
        
        if field_type in numeric_fields:
            return cell_data_type in ['numeric', 'mixed']
        elif field_type in text_fields:
            return cell_data_type in ['text', 'mixed']
        else:
            return True  # Unknown field types accept any data type
    
    def _determine_match_reasons(self, field: DetectedField, cell: CellData,
                               semantic_score: float, context_score: float,
                               proximity_score: float) -> List[str]:
        """Determine reasons why this cell matches the field"""
        reasons = []
        
        if semantic_score > 0.7:
            reasons.append("Strong semantic similarity")
        elif semantic_score > 0.4:
            reasons.append("Moderate semantic similarity")
        
        if context_score > 0.6:
            reasons.append("Found in relevant context")
        
        if proximity_score > 0.5:
            reasons.append("Appropriate location/structure")
        
        # Specific reasons
        field_name = field.original_text.lower()
        cell_text = str(cell.value).lower()
        context_text = ' '.join(cell.context_labels).lower()
        
        if field_name in cell_text:
            reasons.append("Field name found in cell value")
        
        if field_name in context_text:
            reasons.append("Field name found in context")
        
        if field_name in cell.column_name.lower():
            reasons.append("Field name found in column header")
        
        if self._check_data_type_compatibility(field.field_type, cell.data_type):
            reasons.append("Compatible data type")
        
        if cell.confidence > 0.8:
            reasons.append("High-confidence cell")
        
        return reasons
    
    def _generate_resolution_suggestions(self, field: DetectedField, 
                                       scored_matches: List[CellMatch]) -> List[str]:
        """Generate suggestions for field resolution"""
        suggestions = []
        
        if not scored_matches:
            suggestions.append("No potential matches found - check template field name")
            suggestions.append("Consider manual mapping or field name adjustment")
            return suggestions
        
        best_match = scored_matches[0]
        
        if best_match.final_score < 0.4:
            suggestions.append("Low confidence match - manual review recommended")
        
        if best_match.final_score < 0.8 and len(scored_matches) > 1:
            suggestions.append("Multiple similar matches found - review alternatives")
        
        # Type-specific suggestions
        if field.field_type == FieldType.PERSONNEL_NAME:
            if not any("name" in reason.lower() for reason in best_match.match_reasons):
                suggestions.append("Consider checking cells with people's names")
        
        elif field.field_type == FieldType.EXPENSE_AMOUNT:
            if best_match.cell.data_type != 'numeric':
                suggestions.append("Consider checking numeric cells with dollar amounts")
        
        elif field.field_type in [FieldType.NOTES, FieldType.DESCRIPTION]:
            suggestions.append("Check rightmost columns for notes/descriptions")
        
        # Add context-specific suggestions
        if field.grant_specific:
            suggestions.append("Look for grant-specific terminology in context")
        
        return suggestions[:4]  # Limit to top 4 suggestions
    
    def get_resolution_summary(self, resolution_results: List[ResolutionResult]) -> Dict[str, Any]:
        """Generate summary of resolution results"""
        total_fields = len(resolution_results)
        resolved_fields = len([r for r in resolution_results if r.primary_match])
        high_confidence = len([r for r in resolution_results if r.resolution_confidence > 0.8])
        needs_review = len([r for r in resolution_results if r.needs_manual_review])
        
        # Field type breakdown
        field_type_breakdown = defaultdict(int)
        for result in resolution_results:
            field_type_breakdown[result.field.field_type.value] += 1
        
        # Average confidence
        avg_confidence = (
            sum(r.resolution_confidence for r in resolution_results) / total_fields
            if total_fields > 0 else 0
        )
        
        return {
            'total_fields': total_fields,
            'resolved_fields': resolved_fields,
            'high_confidence_resolutions': high_confidence,
            'needs_manual_review': needs_review,
            'average_confidence': avg_confidence,
            'field_type_breakdown': dict(field_type_breakdown),
            'resolution_rate': resolved_fields / total_fields if total_fields > 0 else 0
        }
    
    def export_mapping_report(self, resolution_results: List[ResolutionResult]) -> str:
        """Export detailed mapping report"""
        report = []
        report.append("=== FIELD MAPPING REPORT ===\n")
        
        for i, result in enumerate(resolution_results, 1):
            report.append(f"{i}. Field: {result.field.placeholder}")
            report.append(f"   Type: {result.field.field_type.value}")
            report.append(f"   Confidence: {result.resolution_confidence:.2f}")
            
            if result.primary_match:
                cell = result.primary_match.cell
                report.append(f"   Matched to: {cell.value} (Row {cell.row}, Col {cell.col})")
                report.append(f"   Reasons: {', '.join(result.primary_match.match_reasons)}")
            else:
                report.append("   No match found")
            
            if result.needs_manual_review:
                report.append("   ⚠️  Manual review recommended")
            
            if result.suggestions:
                report.append(f"   Suggestions: {'; '.join(result.suggestions)}")
            
            report.append("")
        
        return "\n".join(report)
    
    def suggest_template_improvements(self, resolution_results: List[ResolutionResult]) -> List[str]:
        """Suggest improvements to template field names"""
        suggestions = []
        
        low_confidence_fields = [r for r in resolution_results if r.resolution_confidence < 0.5]
        
        for result in low_confidence_fields:
            field_name = result.field.original_text
            
            if result.primary_match:
                cell_value = result.primary_match.cell.value
                context_labels = result.primary_match.cell.context_labels
                
                # Suggest better field names based on actual cell content
                if context_labels:
                    best_label = max(context_labels, key=len)  # Use longest context label
                    suggestions.append(f"Consider renaming '{field_name}' to '{best_label}'")
            else:
                suggestions.append(f"Field '{field_name}' could not be matched - consider more specific naming")
        
        return suggestions