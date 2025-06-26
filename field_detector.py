"""
Enhanced Field Detector - AI-powered template field detection with grant-specific intelligence
Focuses on detecting personnel, expenses, and contextual fields in grant templates
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum

class FieldType(Enum):
    """Types of fields commonly found in grant templates"""
    PERSONNEL_NAME = "personnel_name"
    PERSONNEL_TITLE = "personnel_title"
    PERSONNEL_SALARY = "personnel_salary"
    PERSONNEL_EFFORT = "personnel_effort"
    EXPENSE_ITEM = "expense_item"
    EXPENSE_AMOUNT = "expense_amount"
    EXPENSE_CATEGORY = "expense_category"
    DESCRIPTION = "description"
    NOTES = "notes"
    DATE = "date"
    QUANTITY = "quantity"
    RATE = "rate"
    TOTAL = "total"
    YEAR = "year"
    BUDGET_CATEGORY = "budget_category"
    JUSTIFICATION = "justification"
    UNKNOWN = "unknown"

@dataclass
class DetectedField:
    """A field detected in a template"""
    placeholder: str
    original_text: str
    field_type: FieldType
    confidence: float
    context_before: str
    context_after: str
    suggested_mappings: List[str]  # Suggested budget cell mappings
    grant_specific: bool
    position: Tuple[int, int]  # Start and end positions in text

@dataclass
class FieldPattern:
    """Pattern for detecting specific field types"""
    field_type: FieldType
    patterns: List[str]
    context_patterns: List[str]
    confidence_boost: float
    grant_specific: bool = True

class EnhancedFieldDetector:
    """Enhanced field detector with grant proposal intelligence"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.field_patterns = self._initialize_field_patterns()
        self.placeholder_patterns = [
            r'\{([^}]+)\}',           # {field_name}
            r'\[([^\]]+)\]',          # [field_name]
            r'<<([^>]+)>>',           # <<field_name>>
            r'\$\{([^}]+)\}',         # ${field_name}
            r'{{([^}]+)}}',           # {{field_name}}
            r'__([^_]+)__',           # __field_name__
            r'_([A-Z_][A-Z0-9_]+)_',  # _FIELD_NAME_
        ]
        
    def _initialize_field_patterns(self) -> List[FieldPattern]:
        """Initialize patterns for detecting different field types"""
        return [
            # Personnel patterns
            FieldPattern(
                field_type=FieldType.PERSONNEL_NAME,
                patterns=[
                    r'name', r'investigator', r'researcher', r'scientist', r'pi\b', r'co.?pi',
                    r'principal', r'person', r'staff', r'employee', r'faculty', r'director'
                ],
                context_patterns=[
                    r'key personnel', r'project team', r'research team', r'staff list',
                    r'investigators?', r'personnel table', r'team member'
                ],
                confidence_boost=0.3
            ),
            
            FieldPattern(
                field_type=FieldType.PERSONNEL_TITLE,
                patterns=[
                    r'title', r'position', r'role', r'rank', r'appointment', r'designation'
                ],
                context_patterns=[
                    r'job title', r'academic rank', r'professional title', r'position held'
                ],
                confidence_boost=0.3
            ),
            
            FieldPattern(
                field_type=FieldType.PERSONNEL_SALARY,
                patterns=[
                    r'salary', r'wage', r'compensation', r'pay', r'stipend', r'amount',
                    r'annual', r'monthly', r'hourly', r'rate'
                ],
                context_patterns=[
                    r'personnel cost', r'salary cost', r'compensation table', r'pay scale'
                ],
                confidence_boost=0.4
            ),
            
            FieldPattern(
                field_type=FieldType.PERSONNEL_EFFORT,
                patterns=[
                    r'effort', r'percent', r'time', r'fte', r'commitment', r'allocation',
                    r'percentage', r'dedication'
                ],
                context_patterns=[
                    r'percent effort', r'time commitment', r'effort percentage', r'fte allocation'
                ],
                confidence_boost=0.4
            ),
            
            # Expense patterns
            FieldPattern(
                field_type=FieldType.EXPENSE_ITEM,
                patterns=[
                    r'item', r'equipment', r'supply', r'material', r'tool', r'instrument',
                    r'software', r'license', r'subscription', r'service'
                ],
                context_patterns=[
                    r'equipment list', r'supplies needed', r'materials required', 
                    r'itemized', r'line item'
                ],
                confidence_boost=0.3
            ),
            
            FieldPattern(
                field_type=FieldType.EXPENSE_AMOUNT,
                patterns=[
                    r'cost', r'price', r'amount', r'value', r'total', r'sum', r'expense',
                    r'budget', r'dollar', r'usd', r'fee'
                ],
                context_patterns=[
                    r'unit cost', r'total cost', r'estimated cost', r'budget amount',
                    r'expense total'
                ],
                confidence_boost=0.4
            ),
            
            FieldPattern(
                field_type=FieldType.EXPENSE_CATEGORY,
                patterns=[
                    r'category', r'type', r'class', r'group', r'classification',
                    r'equipment', r'travel', r'supplies', r'personnel', r'indirect',
                    r'direct', r'overhead'
                ],
                context_patterns=[
                    r'budget category', r'expense type', r'cost category', r'classification'
                ],
                confidence_boost=0.3
            ),
            
            # Description and notes patterns
            FieldPattern(
                field_type=FieldType.DESCRIPTION,
                patterns=[
                    r'description', r'details', r'specification', r'summary', r'overview',
                    r'explanation', r'info', r'information'
                ],
                context_patterns=[
                    r'item description', r'detailed description', r'brief description'
                ],
                confidence_boost=0.2
            ),
            
            FieldPattern(
                field_type=FieldType.NOTES,
                patterns=[
                    r'notes?', r'comments?', r'remarks?', r'memo', r'additional',
                    r'justification', r'rationale', r'purpose'
                ],
                context_patterns=[
                    r'additional notes', r'comments section', r'remarks field',
                    r'justification text'
                ],
                confidence_boost=0.2
            ),
            
            FieldPattern(
                field_type=FieldType.JUSTIFICATION,
                patterns=[
                    r'justification', r'rationale', r'reason', r'explanation', r'why',
                    r'necessity', r'importance', r'significance'
                ],
                context_patterns=[
                    r'budget justification', r'cost justification', r'expense rationale'
                ],
                confidence_boost=0.4
            ),
            
            # Temporal patterns
            FieldPattern(
                field_type=FieldType.YEAR,
                patterns=[
                    r'year', r'yr', r'annual', r'fy', r'fiscal', r'period', r'term'
                ],
                context_patterns=[
                    r'budget year', r'fiscal year', r'project year', r'year \d+'
                ],
                confidence_boost=0.3
            ),
            
            FieldPattern(
                field_type=FieldType.DATE,
                patterns=[
                    r'date', r'when', r'time', r'start', r'end', r'begin', r'finish',
                    r'deadline', r'due'
                ],
                context_patterns=[
                    r'start date', r'end date', r'project timeline', r'schedule'
                ],
                confidence_boost=0.2
            ),
            
            # Quantitative patterns
            FieldPattern(
                field_type=FieldType.QUANTITY,
                patterns=[
                    r'quantity', r'qty', r'number', r'count', r'units?', r'how.?many',
                    r'amount'
                ],
                context_patterns=[
                    r'quantity needed', r'number of units', r'item count'
                ],
                confidence_boost=0.3
            ),
            
            FieldPattern(
                field_type=FieldType.RATE,
                patterns=[
                    r'rate', r'per', r'each', r'unit', r'hourly', r'daily', r'weekly',
                    r'monthly', r'annually'
                ],
                context_patterns=[
                    r'hourly rate', r'unit rate', r'cost per', r'rate per'
                ],
                confidence_boost=0.3
            ),
            
            FieldPattern(
                field_type=FieldType.TOTAL,
                patterns=[
                    r'total', r'sum', r'grand', r'overall', r'final', r'aggregate',
                    r'combined', r'subtotal'
                ],
                context_patterns=[
                    r'grand total', r'total cost', r'total amount', r'final total'
                ],
                confidence_boost=0.4
            )
        ]
    
    def detect_fields(self, template_text: str) -> List[DetectedField]:
        """Detect all fields in template text"""
        detected_fields = []
        
        # Find all placeholder patterns
        for pattern in self.placeholder_patterns:
            matches = re.finditer(pattern, template_text, re.IGNORECASE)
            
            for match in matches:
                placeholder = match.group(0)
                field_content = match.group(1)
                start_pos = match.start()
                end_pos = match.end()
                
                # Get context around the placeholder
                context_before = self._extract_context(template_text, start_pos, -50)
                context_after = self._extract_context(template_text, end_pos, 50)
                
                # Analyze the field
                field_type, confidence, grant_specific = self._analyze_field(
                    field_content, context_before, context_after
                )
                
                # Generate suggested mappings
                suggested_mappings = self._generate_mapping_suggestions(
                    field_content, field_type, context_before, context_after
                )
                
                detected_field = DetectedField(
                    placeholder=placeholder,
                    original_text=field_content,
                    field_type=field_type,
                    confidence=confidence,
                    context_before=context_before,
                    context_after=context_after,
                    suggested_mappings=suggested_mappings,
                    grant_specific=grant_specific,
                    position=(start_pos, end_pos)
                )
                
                detected_fields.append(detected_field)
        
        # Sort by position in document
        detected_fields.sort(key=lambda x: x.position[0])
        
        self.logger.info(f"Detected {len(detected_fields)} fields in template")
        return detected_fields
    
    def _extract_context(self, text: str, position: int, length: int) -> str:
        """Extract context text around a position"""
        if length > 0:  # Context after
            end_pos = min(position + length, len(text))
            context = text[position:end_pos]
        else:  # Context before
            start_pos = max(0, position + length)
            context = text[start_pos:position]
        
        return context.strip()
    
    def _analyze_field(self, field_content: str, context_before: str, 
                      context_after: str) -> Tuple[FieldType, float, bool]:
        """Analyze a field to determine its type and confidence"""
        field_lower = field_content.lower()
        context_text = (context_before + " " + context_after).lower()
        
        best_match = None
        best_confidence = 0.0
        is_grant_specific = False
        
        # Check each pattern
        for pattern_def in self.field_patterns:
            confidence = 0.0
            
            # Check main patterns
            for pattern in pattern_def.patterns:
                if re.search(pattern, field_lower):
                    confidence += 0.3
                    break
            
            # Check context patterns
            for context_pattern in pattern_def.context_patterns:
                if re.search(context_pattern, context_text):
                    confidence += pattern_def.confidence_boost
                    break
            
            # Additional confidence for exact matches
            if field_lower in [p.replace(r'\b', '').replace('\\', '') 
                              for p in pattern_def.patterns]:
                confidence += 0.2
            
            # Grant-specific boost
            if pattern_def.grant_specific:
                grant_indicators = [
                    'grant', 'proposal', 'nsf', 'nih', 'award', 'funding',
                    'budget', 'personnel', 'investigator'
                ]
                for indicator in grant_indicators:
                    if indicator in context_text:
                        confidence += 0.1
                        is_grant_specific = True
                        break
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = pattern_def.field_type
                if pattern_def.grant_specific:
                    is_grant_specific = True
        
        # Default classification if no pattern matches
        if best_match is None:
            best_match = self._classify_by_name(field_content)
            best_confidence = 0.3
        
        return best_match, min(best_confidence, 1.0), is_grant_specific
    
    def _classify_by_name(self, field_name: str) -> FieldType:
        """Classify field by name when no patterns match"""
        name_lower = field_name.lower()
        
        # Simple heuristics based on common naming conventions
        if any(word in name_lower for word in ['name', 'person', 'who']):
            return FieldType.PERSONNEL_NAME
        elif any(word in name_lower for word in ['title', 'position', 'role']):
            return FieldType.PERSONNEL_TITLE
        elif any(word in name_lower for word in ['cost', 'price', 'amount', 'dollar']):
            return FieldType.EXPENSE_AMOUNT
        elif any(word in name_lower for word in ['item', 'equipment', 'supply']):
            return FieldType.EXPENSE_ITEM
        elif any(word in name_lower for word in ['note', 'comment', 'description']):
            return FieldType.DESCRIPTION
        elif any(word in name_lower for word in ['year', 'date', 'time']):
            return FieldType.YEAR
        else:
            return FieldType.UNKNOWN
    
    def _generate_mapping_suggestions(self, field_content: str, field_type: FieldType,
                                    context_before: str, context_after: str) -> List[str]:
        """Generate suggestions for mapping this field to budget data"""
        suggestions = []
        
        field_lower = field_content.lower()
        context_text = (context_before + " " + context_after).lower()
        
        # Generate suggestions based on field type
        if field_type == FieldType.PERSONNEL_NAME:
            suggestions.extend([
                "Look for cells containing names (e.g., 'John Smith', 'Dr. Johnson')",
                "Check cells near 'investigator', 'PI', or 'researcher' labels",
                "Look for cells with title patterns (Dr., Prof., Mr., Ms.)",
                "Search in personnel or staff sections of budget"
            ])
            
        elif field_type == FieldType.PERSONNEL_TITLE:
            suggestions.extend([
                "Look for job titles (Professor, Research Scientist, etc.)",
                "Check cells near name entries",
                "Look for academic ranks or professional designations",
                "Search for cells containing 'investigator', 'staff', 'specialist'"
            ])
            
        elif field_type == FieldType.PERSONNEL_SALARY:
            suggestions.extend([
                "Look for numeric values in salary/wage columns",
                "Check cells with dollar amounts near personnel names",
                "Look for annual, monthly, or hourly rate values",
                "Search in compensation or pay sections"
            ])
            
        elif field_type == FieldType.PERSONNEL_EFFORT:
            suggestions.extend([
                "Look for percentage values (e.g., 25%, 0.25)",
                "Check cells near 'effort', 'time', or 'FTE' labels",
                "Look for decimal values between 0 and 1",
                "Search in effort allocation sections"
            ])
            
        elif field_type == FieldType.EXPENSE_ITEM:
            suggestions.extend([
                "Look for equipment or supply names",
                "Check item description columns",
                "Look for product names or model numbers",
                "Search in materials or equipment sections"
            ])
            
        elif field_type == FieldType.EXPENSE_AMOUNT:
            suggestions.extend([
                "Look for dollar amounts or costs",
                "Check numeric values in cost columns",
                "Look for price information",
                "Search for budget amounts or totals"
            ])
            
        elif field_type == FieldType.DESCRIPTION or field_type == FieldType.NOTES:
            suggestions.extend([
                "Look in notes or description columns (often rightmost)",
                "Check cells with longer text content",
                "Look for justification or explanation text",
                "Search in comments or details sections"
            ])
            
        elif field_type == FieldType.YEAR:
            suggestions.extend([
                "Look for 4-digit year values (e.g., 2024, 2025)",
                "Check cells with year labels",
                "Look for fiscal year designations",
                "Search in timeline or schedule sections"
            ])
            
        # Add context-specific suggestions
        if 'total' in context_text:
            suggestions.append("Look for sum or total values")
        if 'unit' in context_text:
            suggestions.append("Look for per-unit costs or quantities")
        if 'annual' in context_text:
            suggestions.append("Look for yearly amounts")
        
        # Add field-name specific suggestions
        if field_lower:
            suggestions.append(f"Search for cells containing or near '{field_content}'")
        
        return suggestions[:6]  # Limit to top 6 suggestions
    
    def get_fields_by_type(self, detected_fields: List[DetectedField], 
                          field_type: FieldType) -> List[DetectedField]:
        """Get all detected fields of a specific type"""
        return [field for field in detected_fields if field.field_type == field_type]
    
    def get_high_confidence_fields(self, detected_fields: List[DetectedField], 
                                 min_confidence: float = 0.7) -> List[DetectedField]:
        """Get fields with high confidence scores"""
        return [field for field in detected_fields if field.confidence >= min_confidence]
    
    def get_grant_specific_fields(self, detected_fields: List[DetectedField]) -> List[DetectedField]:
        """Get fields that are specifically related to grant proposals"""
        return [field for field in detected_fields if field.grant_specific]
    
    def generate_field_summary(self, detected_fields: List[DetectedField]) -> Dict[str, Any]:
        """Generate a summary of detected fields"""
        field_type_counts = {}
        for field in detected_fields:
            field_type_name = field.field_type.value
            field_type_counts[field_type_name] = field_type_counts.get(field_type_name, 0) + 1
        
        high_confidence_count = len(self.get_high_confidence_fields(detected_fields))
        grant_specific_count = len(self.get_grant_specific_fields(detected_fields))
        
        return {
            'total_fields': len(detected_fields),
            'field_types': field_type_counts,
            'high_confidence_fields': high_confidence_count,
            'grant_specific_fields': grant_specific_count,
            'average_confidence': sum(f.confidence for f in detected_fields) / len(detected_fields) if detected_fields else 0
        }
    
    def suggest_improvements(self, detected_fields: List[DetectedField]) -> List[str]:
        """Suggest improvements for field detection"""
        suggestions = []
        
        low_confidence_fields = [f for f in detected_fields if f.confidence < 0.5]
        if low_confidence_fields:
            suggestions.append(f"Consider reviewing {len(low_confidence_fields)} low-confidence field(s)")
        
        unknown_fields = self.get_fields_by_type(detected_fields, FieldType.UNKNOWN)
        if unknown_fields:
            suggestions.append(f"Classify {len(unknown_fields)} unknown field type(s) manually")
        
        # Check for common missing field types in grant proposals
        field_types_present = {f.field_type for f in detected_fields}
        important_grant_fields = {
            FieldType.PERSONNEL_NAME, FieldType.PERSONNEL_SALARY, 
            FieldType.EXPENSE_ITEM, FieldType.EXPENSE_AMOUNT
        }
        
        missing_important = important_grant_fields - field_types_present
        if missing_important:
            missing_names = [ft.value for ft in missing_important]
            suggestions.append(f"Consider adding fields for: {', '.join(missing_names)}")
        
        return suggestions