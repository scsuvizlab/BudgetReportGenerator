"""
AI-powered budget cell matching with enhanced string analysis and contextual field detection.
Improved to better handle names, positions, notes, and grant-specific terminology.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from rapidfuzz import fuzz, process
import pandas as pd

@dataclass
class CellContext:
    """Contains contextual information about a cell and its surroundings."""
    value: Any
    row: int
    col: int
    nearby_text: List[str]
    column_header: Optional[str]
    row_header: Optional[str]
    context_type: str  # 'numeric', 'text', 'mixed', 'empty'
    confidence: float = 0.0

@dataclass
class FieldMatch:
    """Represents a matched field with contextual information."""
    field_name: str
    cell_address: str
    value: Any
    context: CellContext
    match_type: str  # 'exact', 'fuzzy', 'contextual', 'keyword'
    confidence: float
    notes: str = ""

class CellResolver:
    """Enhanced cell resolver with improved string analysis and contextual matching."""
    
    def __init__(self, llm_client=None, enable_debug=False):
        self.llm_client = llm_client
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        # Grant-specific terminology patterns
        self.personnel_keywords = {
            'investigator': ['investigator', 'pi', 'principal investigator', 'co-investigator', 'co-pi'],
            'staff': ['staff', 'personnel', 'employee', 'researcher', 'associate'],
            'key_personnel': ['key personnel', 'key staff', 'senior personnel', 'core team'],
            'specialist': ['specialist', 'expert', 'consultant', 'advisor', 'technician'],
            'postdoc': ['postdoc', 'post-doc', 'postdoctoral', 'fellow'],
            'graduate': ['graduate', 'grad student', 'phd student', 'doctoral'],
            'undergraduate': ['undergraduate', 'undergrad', 'student assistant']
        }
        
        self.expense_categories = {
            'equipment': ['equipment', 'instrument', 'hardware', 'computer', 'software'],
            'travel': ['travel', 'conference', 'meeting', 'trip', 'transportation', 'lodging'],
            'supplies': ['supplies', 'materials', 'consumables', 'reagents', 'chemicals'],
            'release_time': ['release time', 'course release', 'teaching release', 'buy-out'],
            'overhead': ['overhead', 'indirect', 'facilities', 'administrative'],
            'fringe': ['fringe', 'benefits', 'insurance', 'retirement']
        }
        
        # Common note column identifiers
        self.note_column_patterns = [
            'notes', 'note', 'description', 'desc', 'comments', 'comment', 
            'details', 'explanation', 'justification', 'remarks', 'memo'
        ]
        
        # Enhanced string analysis patterns
        self.name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',  # Last, First
            r'\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b',  # First M. Last
            r'\bDr\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Dr. Name
        ]
        
        self.title_patterns = [
            r'principal\s+investigator', r'pi\b', r'co-?pi\b',
            r'co-?investigator', r'research\s+(?:scientist|associate|professor)',
            r'post-?doc(?:toral)?(?:\s+(?:fellow|researcher))?',
            r'graduate\s+(?:student|research\s+assistant|assistant)',
            r'undergraduate\s+(?:student|research\s+assistant)',
            r'technician', r'specialist', r'coordinator', r'manager',
            r'professor', r'assistant\s+professor', r'associate\s+professor',
        ]
        
        # Patterns for identifying currency and numeric values
        self.currency_patterns = [
            r'\$[\d,]+\.?\d*',  # $1,234.56
            r'[\d,]+\.?\d*\s*\$',  # 1,234.56 $
            r'[\d,]+\.?\d*\s*dollars?',  # 1234 dollars
            r'USD\s*[\d,]+\.?\d*'  # USD 1234
        ]
        
    def analyze_spreadsheet_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the entire spreadsheet to understand its structure and context."""
        context = {
            'total_rows': len(df),
            'total_cols': len(df.columns),
            'numeric_columns': [],
            'text_columns': [],
            'mixed_columns': [],
            'potential_note_columns': [],
            'personnel_sections': [],
            'expense_sections': [],
            'header_rows': []
        }
        
        # Analyze column types and patterns
        for col_idx in range(len(df.columns)):
            col_data = df.iloc[:, col_idx].dropna()
            col_name = df.columns[col_idx] if col_idx < len(df.columns) else f"Column_{col_idx}"
            
            if col_data.empty:
                continue
                
            numeric_count = sum(pd.api.types.is_numeric_dtype(type(val)) or 
                              self._is_currency_value(str(val)) for val in col_data)
            text_count = sum(isinstance(val, str) and not self._is_currency_value(val) 
                           for val in col_data)
            
            total_count = len(col_data)
            if numeric_count / total_count > 0.8:
                context['numeric_columns'].append((col_idx, col_name))
            elif text_count / total_count > 0.8:
                context['text_columns'].append((col_idx, col_name))
            else:
                context['mixed_columns'].append((col_idx, col_name))
            
            # Check for note columns (typically rightmost text columns)
            if (isinstance(col_name, str) and 
                any(pattern in col_name.lower() for pattern in self.note_column_patterns)):
                context['potential_note_columns'].append((col_idx, col_name))
        
        # Identify personnel and expense sections
        context['personnel_sections'] = self._find_section_patterns(df, self.personnel_keywords)
        context['expense_sections'] = self._find_section_patterns(df, self.expense_categories)
        
        # Find potential header rows
        context['header_rows'] = self._identify_header_rows(df)
        
        return context
    
    def resolve_field_mappings(self, df: pd.DataFrame, template_fields: List[str]) -> List[FieldMatch]:
        """Resolve field mappings with enhanced string analysis and contextual matching."""
        context = self.analyze_spreadsheet_context(df)
        matches = []
        
        # Create cell contexts for all non-empty cells
        cell_contexts = self._create_cell_contexts(df, context)
        
        for field in template_fields:
            try:
                best_matches = self._find_field_matches(field, cell_contexts, df, context)
                matches.extend(best_matches)
            except Exception as e:
                self.logger.warning(f"Field matching failed for field {field}: {e}")
        
        # Post-process matches to resolve conflicts and improve confidence
        matches = self._resolve_match_conflicts(matches)
        
        return matches
    
    def _create_cell_contexts(self, df: pd.DataFrame, spreadsheet_context: Dict) -> List[CellContext]:
        """Create contextual information for each cell in the spreadsheet."""
        contexts = []
        
        for row_idx in range(len(df)):
            for col_idx in range(len(df.columns)):
                value = df.iloc[row_idx, col_idx]
                
                # Skip empty cells
                if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                    continue
                
                # Gather nearby text for context
                nearby_text = self._get_nearby_text(df, row_idx, col_idx, radius=2)
                
                # Determine context type
                context_type = self._determine_context_type(value, nearby_text)
                
                # Get column and row headers
                column_header = df.columns[col_idx] if col_idx < len(df.columns) else None
                row_header = self._get_row_header(df, row_idx)
                
                context = CellContext(
                    value=value,
                    row=row_idx,
                    col=col_idx,
                    nearby_text=nearby_text,
                    column_header=column_header,
                    row_header=row_header,
                    context_type=context_type
                )
                
                contexts.append(context)
        
        return contexts
    
    def _find_field_matches(self, field: str, cell_contexts: List[CellContext], 
                           df: pd.DataFrame, spreadsheet_context: Dict) -> List[FieldMatch]:
        """Find the best matches for a given field using multiple strategies."""
        candidates = []
        
        # Strategy 1: Direct field name matching
        candidates.extend(self._match_by_field_name(field, cell_contexts))
        
        # Strategy 2: Keyword-based matching
        candidates.extend(self._match_by_keywords(field, cell_contexts))
        
        # Strategy 3: Contextual pattern matching
        candidates.extend(self._match_by_context_patterns(field, cell_contexts, spreadsheet_context))
        
        # Strategy 4: Positional and proximity matching
        candidates.extend(self._match_by_position(field, cell_contexts, df))
        
        # Strategy 5: LLM-powered matching (if available)
        if self.llm_client:
            candidates.extend(self._match_with_llm(field, cell_contexts, df))
        
        # Rank and filter candidates
        ranked_candidates = self._rank_candidates(candidates, field)
        
        # Return top matches (usually just 1, but could be multiple for high confidence)
        return ranked_candidates[:3] if ranked_candidates else []
    
    def _match_by_field_name(self, field: str, cell_contexts: List[CellContext]) -> List[FieldMatch]:
        """Match fields by direct name comparison."""
        matches = []
        field_lower = field.lower()
        
        for context in cell_contexts:
            if isinstance(context.value, str):
                value_lower = context.value.lower().strip()
                
                # Exact match
                if value_lower == field_lower:
                    matches.append(FieldMatch(
                        field_name=field,
                        cell_address=f"R{context.row}C{context.col}",
                        value=context.value,
                        context=context,
                        match_type='exact',
                        confidence=0.95,
                        notes="Exact field name match"
                    ))
                
                # High similarity match
                similarity = fuzz.ratio(value_lower, field_lower)
                if similarity > 80:
                    matches.append(FieldMatch(
                        field_name=field,
                        cell_address=f"R{context.row}C{context.col}",
                        value=context.value,
                        context=context,
                        match_type='fuzzy',
                        confidence=similarity / 100.0,
                        notes=f"Fuzzy name match (similarity: {similarity}%)"
                    ))
        
        return matches
    
    def _match_by_keywords(self, field: str, cell_contexts: List[CellContext]) -> List[FieldMatch]:
        """Match fields based on keyword patterns and domain knowledge."""
        matches = []
        field_lower = field.lower()
        
        # Determine field category
        field_category = self._categorize_field(field_lower)
        
        for context in cell_contexts:
            # Check the cell value itself
            match_score = self._calculate_keyword_match_score(field_lower, context, field_category)
            
            if match_score > 0.6:
                matches.append(FieldMatch(
                    field_name=field,
                    cell_address=f"R{context.row}C{context.col}",
                    value=context.value,
                    context=context,
                    match_type='keyword',
                    confidence=match_score,
                    notes=f"Keyword match in category: {field_category}"
                ))
        
        return matches
    
    def _match_by_context_patterns(self, field: str, cell_contexts: List[CellContext], 
                                  spreadsheet_context: Dict) -> List[FieldMatch]:
        """Match fields based on contextual patterns and spreadsheet structure."""
        matches = []
        field_lower = field.lower()
        
        # Look for notes in rightmost columns
        if any(note_term in field_lower for note_term in ['note', 'description', 'comment']):
            for col_idx, col_name in spreadsheet_context['potential_note_columns']:
                for context in cell_contexts:
                    if context.col == col_idx and isinstance(context.value, str) and len(context.value) > 20:
                        matches.append(FieldMatch(
                            field_name=field,
                            cell_address=f"R{context.row}C{context.col}",
                            value=context.value,
                            context=context,
                            match_type='contextual',
                            confidence=0.8,
                            notes="Found in potential notes column"
                        ))
        
        # Look for personnel information in personnel sections
        if any(person_term in field_lower for person_term in ['name', 'investigator', 'staff']):
            for section_info in spreadsheet_context['personnel_sections']:
                start_row, end_row, category = section_info
                for context in cell_contexts:
                    if (start_row <= context.row <= end_row and 
                        isinstance(context.value, str) and 
                        self._looks_like_person_name(context.value)):
                        
                        matches.append(FieldMatch(
                            field_name=field,
                            cell_address=f"R{context.row}C{context.col}",
                            value=context.value,
                            context=context,
                            match_type='contextual',
                            confidence=0.75,
                            notes=f"Found in personnel section: {category}"
                        ))
        
        return matches
    
    def _match_by_position(self, field: str, cell_contexts: List[CellContext], 
                          df: pd.DataFrame) -> List[FieldMatch]:
        """Match fields based on positional relationships and proximity to labels."""
        matches = []
        field_lower = field.lower()
        
        for context in cell_contexts:
            # Look for labels to the left or above
            proximity_score = self._calculate_proximity_score(field_lower, context, df)
            
            if proximity_score > 0.5:
                matches.append(FieldMatch(
                    field_name=field,
                    cell_address=f"R{context.row}C{context.col}",
                    value=context.value,
                    context=context,
                    match_type='contextual',
                    confidence=proximity_score,
                    notes="Found near relevant label"
                ))
        
        return matches
    
    def _match_with_llm(self, field: str, cell_contexts: List[CellContext], 
                       df: pd.DataFrame) -> List[FieldMatch]:
        """Use LLM for sophisticated contextual matching."""
        if not self.llm_client:
            return []
        
        matches = []
        
        # Prepare context for LLM
        context_sample = self._prepare_llm_context(field, cell_contexts[:50], df)
        
        try:
            prompt = f"""Analyze this spreadsheet data to find the best match for the field "{field}".
            
Context: This is a grant budget spreadsheet. Look for:
- People's names and titles near personnel-related fields
- Dollar amounts near budget categories  
- Notes and descriptions in text fields
- Relationships between labels and values

Spreadsheet sample:
{context_sample}

Return the most likely cell coordinates and confidence score (0-1) for the field "{field}".
Focus on string values like names, titles, and descriptions, not just numbers."""
            
            response = self.llm_client.analyze_context(prompt, max_tokens=500)
            
            # Parse LLM response and create matches
            llm_matches = self._parse_llm_response(response, field, cell_contexts)
            matches.extend(llm_matches)
            
        except Exception as e:
            self.logger.warning(f"LLM matching failed for field {field}: {e}")
        
        return matches
    
    def _is_currency_value(self, value_str: str) -> bool:
        """Check if a string represents a currency value."""
        if not isinstance(value_str, str):
            return False
        
        return any(re.search(pattern, value_str.strip()) for pattern in self.currency_patterns)
    
    def _get_nearby_text(self, df: pd.DataFrame, row: int, col: int, radius: int = 2) -> List[str]:
        """Get text from nearby cells for context."""
        nearby_text = []
        
        for r_offset in range(-radius, radius + 1):
            for c_offset in range(-radius, radius + 1):
                if r_offset == 0 and c_offset == 0:
                    continue
                
                target_row = row + r_offset
                target_col = col + c_offset
                
                if (0 <= target_row < len(df) and 0 <= target_col < len(df.columns)):
                    value = df.iloc[target_row, target_col]
                    if isinstance(value, str) and value.strip():
                        nearby_text.append(value.strip())
        
        return nearby_text
    
    def _determine_context_type(self, value: Any, nearby_text: List[str]) -> str:
        """Determine the type of context for a cell."""
        if pd.api.types.is_numeric_dtype(type(value)) or self._is_currency_value(str(value)):
            return 'numeric'
        elif isinstance(value, str):
            if any(self._is_currency_value(text) for text in nearby_text):
                return 'mixed'
            else:
                return 'text'
        else:
            return 'empty'
    
    def _get_row_header(self, df: pd.DataFrame, row_idx: int) -> Optional[str]:
        """Get the leftmost non-empty text cell in a row as a potential row header."""
        for col_idx in range(min(3, len(df.columns))):  # Check first 3 columns
            value = df.iloc[row_idx, col_idx]
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    
    def _find_section_patterns(self, df: pd.DataFrame, keyword_dict: Dict) -> List[Tuple[int, int, str]]:
        """Find sections in the spreadsheet based on keyword patterns."""
        sections = []
        
        for category, keywords in keyword_dict.items():
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    value = df.iloc[row_idx, col_idx]
                    if isinstance(value, str):
                        value_lower = value.lower()
                        if any(keyword in value_lower for keyword in keywords):
                            # Found a section header, determine its range
                            section_end = min(row_idx + 10, len(df))  # Look ahead 10 rows
                            sections.append((row_idx, section_end, category))
                            break
        
        return sections
    
    def _identify_header_rows(self, df: pd.DataFrame) -> List[int]:
        """Identify rows that likely contain headers or section titles."""
        header_rows = []
        
        for row_idx in range(min(5, len(df))):  # Check first 5 rows
            row_data = df.iloc[row_idx].dropna()
            if not row_data.empty:
                text_ratio = sum(isinstance(val, str) for val in row_data) / len(row_data)
                if text_ratio > 0.7:  # Mostly text
                    header_rows.append(row_idx)
        
        return header_rows
    
    def _categorize_field(self, field_lower: str) -> str:
        """Categorize a field based on its name."""
        for category, keywords in {**self.personnel_keywords, **self.expense_categories}.items():
            if any(keyword in field_lower for keyword in keywords):
                return category
        return 'general'
    
    def _calculate_keyword_match_score(self, field_lower: str, context: CellContext, 
                                     field_category: str) -> float:
        """Calculate a match score based on keyword similarity."""
        score = 0.0
        
        # Check the cell value
        if isinstance(context.value, str):
            value_lower = context.value.lower()
            score += fuzz.partial_ratio(field_lower, value_lower) / 100.0 * 0.6
        
        # Check nearby text
        for text in context.nearby_text:
            text_lower = text.lower()
            score += fuzz.partial_ratio(field_lower, text_lower) / 100.0 * 0.2
        
        # Check column header
        if context.column_header and isinstance(context.column_header, str):
            header_lower = context.column_header.lower()
            score += fuzz.partial_ratio(field_lower, header_lower) / 100.0 * 0.3
        
        # Bonus for category match
        if field_category != 'general':
            category_keywords = {**self.personnel_keywords, **self.expense_categories}.get(field_category, [])
            for keyword in category_keywords:
                if isinstance(context.value, str) and keyword in context.value.lower():
                    score += 0.2
                if any(keyword in text.lower() for text in context.nearby_text):
                    score += 0.1
        
        return min(score, 1.0)
    
    def _looks_like_person_name(self, text: str) -> bool:
        """Check if text looks like a person's name."""
        if not isinstance(text, str) or len(text.strip()) < 3:
            return False
        
        # Simple heuristics for person names
        words = text.strip().split()
        if len(words) < 2:
            return False
        
        # Check for common name patterns
        has_capital = any(word[0].isupper() for word in words if word)
        has_reasonable_length = all(2 <= len(word) <= 20 for word in words if word)
        no_numbers = not any(char.isdigit() for char in text)
        
        return has_capital and has_reasonable_length and no_numbers
    
    def _calculate_proximity_score(self, field_lower: str, context: CellContext, 
                                  df: pd.DataFrame) -> float:
        """Calculate proximity score based on nearby labels."""
        score = 0.0
        
        # Check cell to the left
        if context.col > 0:
            left_cell = df.iloc[context.row, context.col - 1]
            if isinstance(left_cell, str):
                score += fuzz.partial_ratio(field_lower, left_cell.lower()) / 100.0 * 0.4
        
        # Check cell above
        if context.row > 0:
            above_cell = df.iloc[context.row - 1, context.col]
            if isinstance(above_cell, str):
                score += fuzz.partial_ratio(field_lower, above_cell.lower()) / 100.0 * 0.3
        
        # Check column header
        if context.column_header and isinstance(context.column_header, str):
            score += fuzz.partial_ratio(field_lower, context.column_header.lower()) / 100.0 * 0.3
        
        return min(score, 1.0)
    
    def _prepare_llm_context(self, field: str, contexts: List[CellContext], 
                           df: pd.DataFrame) -> str:
        """Prepare context information for LLM analysis."""
        context_lines = []
        context_lines.append(f"Looking for field: {field}")
        context_lines.append(f"Spreadsheet size: {len(df)} rows x {len(df.columns)} columns")
        context_lines.append("\nSample cells with context:")
        
        for i, ctx in enumerate(contexts[:20]):  # Limit sample size
            context_lines.append(f"Row {ctx.row}, Col {ctx.col}: '{ctx.value}'")
            if ctx.column_header:
                context_lines.append(f"  Column header: '{ctx.column_header}'")
            if ctx.nearby_text:
                context_lines.append(f"  Nearby: {ctx.nearby_text[:3]}")
        
        return "\n".join(context_lines)
    
    def _parse_llm_response(self, response: str, field: str, 
                           contexts: List[CellContext]) -> List[FieldMatch]:
        """Parse LLM response and create field matches."""
        matches = []
        
        # Simple parsing - look for row/column coordinates and confidence
        lines = response.split('\n')
        for line in lines:
            if 'row' in line.lower() and 'col' in line.lower():
                # Extract coordinates and confidence
                try:
                    # This would need more sophisticated parsing based on actual LLM response format
                    row_match = re.search(r'row\s*(\d+)', line.lower())
                    col_match = re.search(r'col\s*(\d+)', line.lower())
                    conf_match = re.search(r'confidence[:\s]*([\d.]+)', line.lower())
                    
                    if row_match and col_match:
                        row_idx = int(row_match.group(1))
                        col_idx = int(col_match.group(1))
                        confidence = float(conf_match.group(1)) if conf_match else 0.7
                        
                        # Find the corresponding context
                        for ctx in contexts:
                            if ctx.row == row_idx and ctx.col == col_idx:
                                matches.append(FieldMatch(
                                    field_name=field,
                                    cell_address=f"R{row_idx}C{col_idx}",
                                    value=ctx.value,
                                    context=ctx,
                                    match_type='llm',
                                    confidence=confidence,
                                    notes="LLM-powered match"
                                ))
                                break
                except (ValueError, AttributeError):
                    continue
        
        return matches
    
    def _rank_candidates(self, candidates: List[FieldMatch], field: str) -> List[FieldMatch]:
        """Rank and filter candidates based on confidence and other factors."""
        if not candidates:
            return []
        
        # Remove duplicates (same cell address)
        unique_candidates = {}
        for candidate in candidates:
            if candidate.cell_address not in unique_candidates:
                unique_candidates[candidate.cell_address] = candidate
            else:
                # Keep the one with higher confidence
                if candidate.confidence > unique_candidates[candidate.cell_address].confidence:
                    unique_candidates[candidate.cell_address] = candidate
        
        # Sort by confidence
        ranked = sorted(unique_candidates.values(), key=lambda x: x.confidence, reverse=True)
        
        # Filter by minimum confidence threshold
        return [match for match in ranked if match.confidence > 0.3]
    
    def _resolve_match_conflicts(self, matches: List[FieldMatch]) -> List[FieldMatch]:
        """Resolve conflicts when multiple fields map to the same cell."""
        cell_to_matches = {}
        
        # Group matches by cell address
        for match in matches:
            if match.cell_address not in cell_to_matches:
                cell_to_matches[match.cell_address] = []
            cell_to_matches[match.cell_address].append(match)
        
        resolved_matches = []
        
        for cell_address, cell_matches in cell_to_matches.items():
            if len(cell_matches) == 1:
                resolved_matches.append(cell_matches[0])
            else:
                # Multiple fields want the same cell - pick the highest confidence
                best_match = max(cell_matches, key=lambda x: x.confidence)
                best_match.notes += f" (resolved conflict with {len(cell_matches)-1} other fields)"
                resolved_matches.append(best_match)
        
        return resolved_matches

    def analyze_text_content(self, text: str) -> Dict[str, Any]:
        """Enhanced text analysis to determine content type and characteristics."""
        if not isinstance(text, str) or not text.strip():
            return {'type': 'empty', 'confidence': 0.0, 'features': {}}
        
        text = text.strip()
        analysis = {'type': 'text', 'confidence': 0.5, 'features': {}}
        
        # Check if it's a person name
        if self._is_person_name(text):
            analysis.update({'type': 'person_name', 'confidence': 0.85, 'features': {'name_patterns': True}})
        
        # Check if it's a job title
        elif self._is_job_title(text):
            analysis.update({'type': 'job_title', 'confidence': 0.80, 'features': {'title_patterns': True}})
        
        # Check if it's a long description (likely notes)
        elif len(text) > 50:
            analysis.update({'type': 'description', 'confidence': 0.75, 'features': {'long_text': True, 'length': len(text)}})
        
        # Check for grant terminology
        grant_terms = self._find_grant_terminology(text)
        if grant_terms:
            analysis['features']['grant_terms'] = grant_terms
            analysis['confidence'] += 0.1
        
        return analysis
    
    def _is_person_name(self, text: str) -> bool:
        """Enhanced person name detection using multiple patterns."""
        if not isinstance(text, str) or len(text.strip()) < 3:
            return False
        
        # Check against name patterns
        for pattern in self.name_patterns:
            if re.search(pattern, text):
                return True
        
        # Additional heuristics
        words = text.strip().split()
        if len(words) < 2 or len(words) > 4:
            return False
        
        # Names typically start with capital letters
        if not all(word[0].isupper() for word in words if word):
            return False
        
        # Names shouldn't contain numbers
        if any(char.isdigit() for char in text):
            return False
        
        return True
    
    def _is_job_title(self, text: str) -> bool:
        """Enhanced job title detection using grant-specific patterns."""
        if not isinstance(text, str) or len(text.strip()) < 3:
            return False
        
        text_lower = text.lower().strip()
        
        # Check against title patterns
        for pattern in self.title_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for general job title indicators
        title_words = ['director', 'manager', 'coordinator', 'administrator', 'supervisor']
        return any(word in text_lower for word in title_words)
    
    def _find_grant_terminology(self, text: str) -> List[str]:
        """Find grant-specific terminology in text."""
        text_lower = text.lower()
        matches = []
        
        # Check all category keywords
        for category, keywords in {**self.personnel_keywords, **self.expense_categories}.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matches.append(keyword)
        
        return matches
    
    def find_notes_columns(self, df: pd.DataFrame) -> List[Tuple[int, str, float]]:
        """Find columns that likely contain notes or descriptions."""
        potential_notes = []
        
        # Check column headers
        for col_idx, col_name in enumerate(df.columns):
            if isinstance(col_name, str):
                header_lower = col_name.lower().strip()
                for pattern in self.note_column_patterns:
                    if pattern in header_lower:
                        potential_notes.append((col_idx, col_name, 0.9))
                        break
        
        # Check rightmost columns with mostly text
        if len(df.columns) > 3:  # Only if there are enough columns
            rightmost_cols = list(range(len(df.columns) - 3, len(df.columns)))
            
            for col_idx in rightmost_cols:
                if col_idx < len(df.columns):
                    col_data = df.iloc[:, col_idx].dropna()
                    if len(col_data) > 0:
                        text_ratio = sum(isinstance(val, str) and len(str(val)) > 20 
                                       for val in col_data) / len(col_data)
                        
                        if text_ratio > 0.6:  # Mostly long text
                            col_name = df.columns[col_idx] if col_idx < len(df.columns) else f"Column_{col_idx}"
                            # Check if not already found
                            if not any(idx == col_idx for idx, _, _ in potential_notes):
                                potential_notes.append((col_idx, str(col_name), text_ratio * 0.7))
        
        return potential_notes
    
    def extract_personnel_from_context(self, contexts: List[CellContext]) -> List[Dict[str, Any]]:
        """Extract personnel information from cell contexts."""
        personnel_entries = []
        
        # Group contexts by row to find related information
        row_groups = {}
        for context in contexts:
            if context.row not in row_groups:
                row_groups[context.row] = []
            row_groups[context.row].append(context)
        
        for row_idx, row_contexts in row_groups.items():
            # Look for names in this row
            names = []
            titles = []
            costs = []
            notes = []
            
            for context in row_contexts:
                if context.context_type == 'text' and isinstance(context.value, str):
                    text_analysis = self.analyze_text_content(context.value)
                    
                    if text_analysis['type'] == 'person_name':
                        names.append({
                            'value': context.value,
                            'location': f"R{context.row}C{context.col}",
                            'confidence': text_analysis['confidence']
                        })
                    elif text_analysis['type'] == 'job_title':
                        titles.append({
                            'value': context.value,
                            'location': f"R{context.row}C{context.col}",
                            'confidence': text_analysis['confidence']
                        })
                    elif text_analysis['type'] == 'description':
                        notes.append({
                            'value': context.value,
                            'location': f"R{context.row}C{context.col}",
                            'confidence': text_analysis['confidence']
                        })
                
                elif context.context_type in ['numeric', 'currency']:
                    costs.append({
                        'value': context.value,
                        'location': f"R{context.row}C{context.col}",
                        'type': context.context_type
                    })
            
            # Create personnel entries for rows with names
            for name_info in names:
                entry = {
                    'name': name_info['value'],
                    'name_location': name_info['location'],
                    'name_confidence': name_info['confidence'],
                    'row': row_idx
                }
                
                # Add associated title if found
                if titles:
                    best_title = max(titles, key=lambda x: x['confidence'])
                    entry['title'] = best_title['value']
                    entry['title_location'] = best_title['location']
                
                # Add associated costs if found
                if costs:
                    entry['associated_costs'] = costs
                
                # Add notes if found
                if notes:
                    entry['notes'] = '; '.join(note['value'] for note in notes)
                
                personnel_entries.append(entry)
        
        return personnel_entries