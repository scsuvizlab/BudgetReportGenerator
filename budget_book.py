"""
Enhanced Budget Book Parser - Improved text and context extraction
Focuses on capturing names, titles, descriptions, and notes alongside numeric values
"""

import pandas as pd
import openpyxl
from openpyxl import load_workbook
import re
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CellData:
    """Enhanced cell data structure to capture both numeric and text information"""
    value: Any
    raw_value: Any
    row: int
    col: int
    column_name: str
    data_type: str  # 'numeric', 'text', 'date', 'mixed'
    confidence: float
    context_labels: List[str]  # Labels found nearby
    notes: str = ""
    formatted_value: str = ""

@dataclass 
class GrantContext:
    """Context information specific to grant proposals"""
    personnel_indicators: List[str]
    expense_categories: List[str]
    note_indicators: List[str]
    title_patterns: List[str]

class EnhancedBudgetBook:
    """Enhanced budget parser that captures text context and grant-specific information"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.workbook = None
        self.sheets_data = {}
        self.all_cells = []
        self.grant_context = self._initialize_grant_context()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_grant_context(self) -> GrantContext:
        """Initialize grant-specific terminology and patterns"""
        return GrantContext(
            personnel_indicators=[
                'investigator', 'pi', 'principal investigator', 'co-investigator', 'co-pi',
                'staff', 'personnel', 'employee', 'researcher', 'scientist', 'analyst',
                'key personnel', 'project director', 'coordinator', 'manager',
                'specialist', 'technician', 'assistant', 'associate', 'fellow',
                'postdoc', 'graduate student', 'undergraduate', 'intern',
                'faculty', 'professor', 'instructor', 'lecturer'
            ],
            expense_categories=[
                'equipment', 'supplies', 'travel', 'publication', 'communication',
                'release time', 'salary', 'wage', 'benefit', 'fringe', 'overhead',
                'indirect', 'direct', 'consultant', 'contractor', 'subcontract',
                'material', 'software', 'license', 'subscription', 'training',
                'conference', 'workshop', 'meeting', 'lodging', 'transportation',
                'per diem', 'meal', 'registration', 'fee'
            ],
            note_indicators=[
                'notes', 'note', 'description', 'details', 'explanation', 'comment',
                'remarks', 'memo', 'justification', 'rationale', 'purpose',
                'objective', 'goal', 'scope', 'deliverable', 'milestone'
            ],
            title_patterns=[
                r'\b(dr\.?|prof\.?|professor|mr\.?|ms\.?|mrs\.?)\s+\w+',
                r'\w+\s+(investigator|researcher|scientist|analyst|specialist)',
                r'(project\s+)?(director|manager|coordinator|lead)',
                r'(senior|junior|lead|principal|associate|assistant)\s+\w+'
            ]
        )
    
    def load_budget_file(self) -> Dict[str, Any]:
        """Load and parse budget file with enhanced text extraction"""
        try:
            if self.file_path.suffix.lower() in ['.xlsx', '.xls']:
                return self._load_excel_file()
            elif self.file_path.suffix.lower() == '.csv':
                return self._load_csv_file()
            else:
                raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        except Exception as e:
            self.logger.error(f"Error loading budget file: {str(e)}")
            raise

    def _load_excel_file(self) -> Dict[str, Any]:
        """Load Excel file with enhanced context extraction"""
        self.workbook = load_workbook(self.file_path, data_only=False)
        
        for sheet_name in self.workbook.sheetnames:
            sheet = self.workbook[sheet_name]
            sheet_data = self._extract_sheet_data(sheet, sheet_name)
            self.sheets_data[sheet_name] = sheet_data
            
        return {
            'sheets': self.sheets_data,
            'all_cells': self.all_cells,
            'file_type': 'excel',
            'summary': self._generate_summary()
        }
    
    def _load_csv_file(self) -> Dict[str, Any]:
        """Load CSV file with enhanced context extraction"""
        df = pd.read_csv(self.file_path)
        sheet_data = self._extract_dataframe_data(df, 'Sheet1')
        self.sheets_data['Sheet1'] = sheet_data
        
        return {
            'sheets': self.sheets_data,
            'all_cells': self.all_cells,
            'file_type': 'csv',
            'summary': self._generate_summary()
        }
    
    def _extract_sheet_data(self, sheet, sheet_name: str) -> Dict[str, Any]:
        """Extract data from Excel sheet with enhanced context awareness"""
        sheet_cells = []
        max_row = sheet.max_row
        max_col = sheet.max_column
        
        # Build a grid for context analysis
        cell_grid = {}
        for row in range(1, max_row + 1):
            for col in range(1, max_col + 1):
                cell = sheet.cell(row=row, column=col)
                if cell.value is not None:
                    cell_grid[(row, col)] = cell
        
        # Extract cells with enhanced context
        for (row, col), cell in cell_grid.items():
            cell_data = self._analyze_cell_with_context(cell, row, col, cell_grid, sheet)
            sheet_cells.append(cell_data)
            self.all_cells.append(cell_data)
        
        # Identify potential notes column
        notes_column = self._identify_notes_column(sheet_cells, max_col)
        
        return {
            'name': sheet_name,
            'cells': sheet_cells,
            'max_row': max_row,
            'max_col': max_col,
            'notes_column': notes_column,
            'personnel_entries': self._extract_personnel_entries(sheet_cells),
            'expense_categories': self._extract_expense_categories(sheet_cells),
            'contextual_groups': self._group_related_cells(sheet_cells)
        }
    
    def _extract_dataframe_data(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Extract data from pandas DataFrame with context awareness"""
        sheet_cells = []
        max_row, max_col = df.shape
        
        # Convert DataFrame to cell-like structure for analysis
        for row_idx in range(len(df)):
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                if pd.notna(value):
                    cell_data = self._analyze_dataframe_cell(
                        value, row_idx + 2, col_idx + 1, col_name, df, row_idx, col_idx
                    )
                    sheet_cells.append(cell_data)
                    self.all_cells.append(cell_data)
        
        # Add header row
        for col_idx, col_name in enumerate(df.columns):
            cell_data = CellData(
                value=col_name,
                raw_value=col_name,
                row=1,
                col=col_idx + 1,
                column_name=col_name,
                data_type='text',
                confidence=1.0,
                context_labels=[],
                formatted_value=str(col_name)
            )
            sheet_cells.append(cell_data)
            self.all_cells.append(cell_data)
        
        notes_column = self._identify_notes_column(sheet_cells, max_col)
        
        return {
            'name': sheet_name,
            'cells': sheet_cells,
            'max_row': max_row + 1,  # +1 for header
            'max_col': max_col,
            'notes_column': notes_column,
            'personnel_entries': self._extract_personnel_entries(sheet_cells),
            'expense_categories': self._extract_expense_categories(sheet_cells),
            'contextual_groups': self._group_related_cells(sheet_cells)
        }
    
    def _analyze_cell_with_context(self, cell, row: int, col: int, cell_grid: Dict, sheet) -> CellData:
        """Analyze individual cell with surrounding context"""
        value = cell.value
        raw_value = value
        
        # Determine data type
        data_type = self._determine_data_type(value)
        
        # Get column name (from first row if available)
        column_name = ""
        if (1, col) in cell_grid:
            column_name = str(cell_grid[(1, col)].value or "")
        
        # Find context labels (nearby text)
        context_labels = self._find_nearby_labels(row, col, cell_grid)
        
        # Calculate confidence based on context
        confidence = self._calculate_confidence(value, data_type, context_labels)
        
        # Get formatted value
        formatted_value = self._format_cell_value(cell, value)
        
        # Extract notes if this appears to be a notes cell
        notes = self._extract_notes(value, context_labels)
        
        return CellData(
            value=value,
            raw_value=raw_value,
            row=row,
            col=col,
            column_name=column_name,
            data_type=data_type,
            confidence=confidence,
            context_labels=context_labels,
            notes=notes,
            formatted_value=formatted_value
        )
    
    def _analyze_dataframe_cell(self, value, row: int, col: int, col_name: str, 
                              df: pd.DataFrame, df_row: int, df_col: int) -> CellData:
        """Analyze DataFrame cell with context"""
        data_type = self._determine_data_type(value)
        
        # Find context from nearby cells
        context_labels = self._find_dataframe_context(df, df_row, df_col)
        context_labels.append(col_name)  # Column name is always relevant context
        
        confidence = self._calculate_confidence(value, data_type, context_labels)
        notes = self._extract_notes(value, context_labels)
        
        return CellData(
            value=value,
            raw_value=value,
            row=row,
            col=col,
            column_name=col_name,
            data_type=data_type,
            confidence=confidence,
            context_labels=context_labels,
            notes=notes,
            formatted_value=str(value)
        )
    
    def _determine_data_type(self, value) -> str:
        """Determine the type of data in a cell"""
        if value is None:
            return 'empty'
        
        if isinstance(value, (int, float)):
            return 'numeric'
        
        if isinstance(value, str):
            # Check if string contains numbers
            if re.search(r'\d', value):
                if re.match(r'^\$?[\d,]+\.?\d*$', value.strip()):
                    return 'numeric'
                else:
                    return 'mixed'
            else:
                return 'text'
        
        # Handle dates and other types
        return 'other'
    
    def _find_nearby_labels(self, row: int, col: int, cell_grid: Dict) -> List[str]:
        """Find text labels near a given cell"""
        labels = []
        
        # Search pattern: left, above, diagonal
        search_positions = [
            (row, col-1), (row, col-2), (row, col-3),  # Left
            (row-1, col), (row-2, col), (row-3, col),  # Above
            (row-1, col-1), (row-1, col-2),           # Diagonal
            (row, col+1), (row, col+2)                # Right (for notes)
        ]
        
        for search_row, search_col in search_positions:
            if (search_row, search_col) in cell_grid:
                nearby_cell = cell_grid[(search_row, search_col)]
                if nearby_cell.value and isinstance(nearby_cell.value, str):
                    label = str(nearby_cell.value).strip()
                    if label and len(label) > 1:  # Avoid single characters
                        labels.append(label)
        
        return labels
    
    def _find_dataframe_context(self, df: pd.DataFrame, row: int, col: int) -> List[str]:
        """Find contextual labels in DataFrame"""
        labels = []
        
        # Check cells to the left
        for i in range(max(0, col-3), col):
            if i < len(df.columns):
                value = df.iloc[row, i] if row < len(df) else None
                if pd.notna(value) and isinstance(value, str):
                    labels.append(str(value).strip())
        
        # Check cells above
        for i in range(max(0, row-3), row):
            if i < len(df):
                value = df.iloc[i, col]
                if pd.notna(value) and isinstance(value, str):
                    labels.append(str(value).strip())
        
        return labels
    
    def _calculate_confidence(self, value, data_type: str, context_labels: List[str]) -> float:
        """Calculate confidence score based on value and context"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for numeric values with financial context
        if data_type == 'numeric':
            confidence += 0.3
            
        # Check for grant-specific context
        context_text = ' '.join(context_labels).lower()
        
        for indicator in self.grant_context.personnel_indicators:
            if indicator in context_text:
                confidence += 0.2
                break
        
        for category in self.grant_context.expense_categories:
            if category in context_text:
                confidence += 0.2
                break
        
        # Boost confidence for text that looks like names or titles
        if data_type in ['text', 'mixed'] and value:
            value_str = str(value).lower()
            for pattern in self.grant_context.title_patterns:
                if re.search(pattern, value_str):
                    confidence += 0.3
                    break
        
        return min(confidence, 1.0)
    
    def _format_cell_value(self, cell, value) -> str:
        """Format cell value preserving original formatting"""
        if hasattr(cell, 'number_format') and cell.number_format:
            # Try to preserve Excel formatting
            return str(value)
        return str(value) if value is not None else ""
    
    def _extract_notes(self, value, context_labels: List[str]) -> str:
        """Extract notes from cell value and context"""
        if not value:
            return ""
        
        value_str = str(value)
        context_text = ' '.join(context_labels).lower()
        
        # Check if this looks like a notes field
        for indicator in self.grant_context.note_indicators:
            if indicator in context_text:
                return value_str
        
        # Check if the value itself looks like a note (longer text)
        if len(value_str) > 20 and not re.match(r'^\$?[\d,]+\.?\d*$', value_str):
            return value_str
        
        return ""
    
    def _identify_notes_column(self, cells: List[CellData], max_col: int) -> Optional[int]:
        """Identify which column likely contains notes/descriptions"""
        # Check rightmost columns first
        for col in range(max_col, 0, -1):
            col_cells = [cell for cell in cells if cell.col == col]
            
            # Check header for notes indicators
            header_cell = next((cell for cell in col_cells if cell.row == 1), None)
            if header_cell and header_cell.value:
                header_text = str(header_cell.value).lower()
                for indicator in self.grant_context.note_indicators:
                    if indicator in header_text:
                        return col
            
            # Check if column contains mostly text
            text_count = sum(1 for cell in col_cells if cell.data_type == 'text' and len(str(cell.value)) > 10)
            if text_count > len(col_cells) * 0.6:  # More than 60% are substantial text
                return col
        
        return None
    
    def _extract_personnel_entries(self, cells: List[CellData]) -> List[Dict[str, Any]]:
        """Extract personnel-related entries"""
        personnel = []
        
        for cell in cells:
            context_text = ' '.join(cell.context_labels).lower()
            cell_text = str(cell.value).lower() if cell.value else ""
            
            # Check for personnel indicators
            is_personnel = False
            for indicator in self.grant_context.personnel_indicators:
                if indicator in context_text or indicator in cell_text:
                    is_personnel = True
                    break
            
            # Check for name patterns
            if cell.data_type == 'text' and cell.value:
                for pattern in self.grant_context.title_patterns:
                    if re.search(pattern, str(cell.value), re.IGNORECASE):
                        is_personnel = True
                        break
            
            if is_personnel:
                personnel.append({
                    'cell': cell,
                    'type': 'personnel',
                    'indicators_found': [ind for ind in self.grant_context.personnel_indicators 
                                       if ind in context_text or ind in cell_text]
                })
        
        return personnel
    
    def _extract_expense_categories(self, cells: List[CellData]) -> List[Dict[str, Any]]:
        """Extract expense category entries"""
        expenses = []
        
        for cell in cells:
            context_text = ' '.join(cell.context_labels).lower()
            cell_text = str(cell.value).lower() if cell.value else ""
            
            for category in self.grant_context.expense_categories:
                if category in context_text or category in cell_text:
                    expenses.append({
                        'cell': cell,
                        'category': category,
                        'type': 'expense'
                    })
                    break
        
        return expenses
    
    def _group_related_cells(self, cells: List[CellData]) -> List[Dict[str, Any]]:
        """Group cells that appear to be related"""
        groups = []
        processed_cells = set()
        
        for cell in cells:
            if id(cell) in processed_cells:
                continue
            
            # Find cells in the same row that might be related
            row_cells = [c for c in cells if c.row == cell.row and abs(c.col - cell.col) <= 3]
            
            if len(row_cells) > 1:
                # Check if they form a logical group
                has_label = any(c.data_type == 'text' for c in row_cells)
                has_value = any(c.data_type in ['numeric', 'mixed'] for c in row_cells)
                
                if has_label and has_value:
                    groups.append({
                        'type': 'row_group',
                        'cells': row_cells,
                        'primary_cell': cell
                    })
                    
                    for c in row_cells:
                        processed_cells.add(id(c))
        
        return groups
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of extracted data"""
        total_cells = len(self.all_cells)
        numeric_cells = len([c for c in self.all_cells if c.data_type == 'numeric'])
        text_cells = len([c for c in self.all_cells if c.data_type == 'text'])
        high_confidence = len([c for c in self.all_cells if c.confidence > 0.8])
        
        return {
            'total_cells': total_cells,
            'numeric_cells': numeric_cells,
            'text_cells': text_cells,
            'high_confidence_cells': high_confidence,
            'sheets_processed': len(self.sheets_data),
            'personnel_entries': sum(len(sheet['personnel_entries']) for sheet in self.sheets_data.values()),
            'expense_entries': sum(len(sheet['expense_categories']) for sheet in self.sheets_data.values())
        }

    def get_cells_by_type(self, data_type: str) -> List[CellData]:
        """Get all cells of a specific data type"""
        return [cell for cell in self.all_cells if cell.data_type == data_type]
    
    def get_high_confidence_cells(self, min_confidence: float = 0.8) -> List[CellData]:
        """Get cells with high confidence scores"""
        return [cell for cell in self.all_cells if cell.confidence >= min_confidence]
    
    def search_cells_by_context(self, search_terms: List[str]) -> List[CellData]:
        """Search for cells based on context labels"""
        matching_cells = []
        search_terms_lower = [term.lower() for term in search_terms]
        
        for cell in self.all_cells:
            context_text = ' '.join(cell.context_labels).lower()
            cell_text = str(cell.value).lower() if cell.value else ""
            
            for term in search_terms_lower:
                if term in context_text or term in cell_text:
                    matching_cells.append(cell)
                    break
        
        return matching_cells