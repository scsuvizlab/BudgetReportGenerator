"""
Budget Book Parser and Representation

Handles parsing of Excel/CSV budget files and extracts numeric values
with their associated labels and years.
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import openpyxl
from openpyxl import load_workbook
import logging

logger = logging.getLogger(__name__)


@dataclass
class BudgetCell:
    """Represents a budget value found in the spreadsheet."""
    sheet: str
    row: int
    col: int
    label: str
    year: Optional[int]
    value: float
    raw_value: Any  # Original cell value
    confidence: float  # How confident we are in this extraction
    context: str  # Surrounding cells for validation


@dataclass
class BudgetBook:
    """Canonical representation of budget spreadsheet data."""
    source_path: Path
    sheets: List[str]
    cells: List[BudgetCell]
    
    def find_by_label(self, label: str, year: Optional[int] = None) -> List[BudgetCell]:
        """Find budget cells matching a label pattern."""
        matches = []
        label_lower = label.lower().replace('_', ' ')
        
        for cell in self.cells:
            cell_label_lower = cell.label.lower()
            
            # Check for exact match first
            if cell_label_lower == label_lower:
                if year is None or cell.year == year:
                    matches.append(cell)
                    continue
            
            # Check for partial matches
            if self._labels_match(label_lower, cell_label_lower):
                if year is None or cell.year == year:
                    matches.append(cell)
        
        # Sort by confidence, highest first
        return sorted(matches, key=lambda x: x.confidence, reverse=True)
    
    def _labels_match(self, target: str, candidate: str) -> bool:
        """Check if two labels are similar enough to be considered a match."""
        from rapidfuzz import fuzz
        
        # Normalize both labels
        target_words = set(re.findall(r'\w+', target.lower()))
        candidate_words = set(re.findall(r'\w+', candidate.lower()))
        
        # Check word overlap
        if len(target_words & candidate_words) >= min(len(target_words), 2):
            return True
        
        # Check string similarity
        similarity = fuzz.ratio(target, candidate)
        return similarity > 70
    
    def get_unique_labels(self) -> List[str]:
        """Get list of unique labels found in the budget."""
        labels = set(cell.label for cell in self.cells)
        return sorted(list(labels))
    
    def get_years(self) -> List[int]:
        """Get list of years found in the budget."""
        years = set(cell.year for cell in self.cells if cell.year is not None)
        return sorted(list(years))


class BudgetParser:
    """Parses budget spreadsheets and extracts labeled numeric values."""
    
    # Common year patterns
    YEAR_PATTERNS = [
        r'20\d{2}',  # 2024, 2025, etc.
        r'FY\s*20\d{2}',  # FY 2024
        r'Year\s*\d+',  # Year 1, Year 2
    ]
    
    # Common budget label patterns
    BUDGET_PATTERNS = [
        r'salary|wage|compensation|personnel',
        r'equipment|instrument|hardware',
        r'travel|transportation',
        r'supplies|materials|consumables',
        r'indirect|overhead|f&a',
        r'total|sum|subtotal',
        r'benefits|fringe',
    ]
    
    def __init__(self):
        self.year_regex = '|'.join(f'({pattern})' for pattern in self.YEAR_PATTERNS)
        self.budget_regex = '|'.join(f'({pattern})' for pattern in self.BUDGET_PATTERNS)
    
    def parse_file(self, file_path: Path) -> BudgetBook:
        """Parse a budget file and return BudgetBook."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Budget file not found: {file_path}")
        
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            return self._parse_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._parse_csv(file_path)
        else:
            raise ValueError(f"Unsupported budget format: {file_path.suffix}")
    
    def _parse_excel(self, file_path: Path) -> BudgetBook:
        """Parse an Excel file."""
        try:
            workbook = load_workbook(file_path, data_only=True)
            all_cells = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_cells = self._extract_sheet_data(sheet, sheet_name)
                all_cells.extend(sheet_cells)
            
            return BudgetBook(
                source_path=file_path,
                sheets=workbook.sheetnames,
                cells=all_cells
            )
            
        except Exception as e:
            logger.error(f"Error parsing Excel file {file_path}: {e}")
            raise
    
    def _parse_csv(self, file_path: Path) -> BudgetBook:
        """Parse a CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to a format similar to Excel sheet
            cells = []
            for row_idx, row in df.iterrows():
                for col_idx, value in enumerate(row):
                    if pd.notna(value) and self._is_numeric_value(value):
                        # Look for label in same row or column header
                        label = self._find_label_for_csv_cell(df, row_idx, col_idx)
                        year = self._extract_year_from_context(str(value) + " " + label)
                        
                        cell = BudgetCell(
                            sheet="Sheet1",
                            row=row_idx + 1,  # 1-indexed like Excel
                            col=col_idx + 1,
                            label=label,
                            year=year,
                            value=float(value),
                            raw_value=value,
                            confidence=0.8,  # Lower confidence for CSV
                            context=self._get_csv_context(df, row_idx, col_idx)
                        )
                        cells.append(cell)
            
            return BudgetBook(
                source_path=file_path,
                sheets=["Sheet1"],
                cells=cells
            )
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}")
            raise
    
    def _extract_sheet_data(self, sheet, sheet_name: str) -> List[BudgetCell]:
        """Extract budget data from a single Excel sheet."""
        cells = []
        
        # Scan all cells looking for numeric values
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None and self._is_numeric_value(cell.value):
                    # Found a numeric value, now find its label
                    label = self._find_label_for_cell(sheet, cell.row, cell.column)
                    year = self._extract_year_from_context(
                        self._get_cell_context(sheet, cell.row, cell.column)
                    )
                    
                    budget_cell = BudgetCell(
                        sheet=sheet_name,
                        row=cell.row,
                        col=cell.column,
                        label=label,
                        year=year,
                        value=float(cell.value),
                        raw_value=cell.value,
                        confidence=self._calculate_confidence(sheet, cell.row, cell.column),
                        context=self._get_cell_context(sheet, cell.row, cell.column)
                    )
                    cells.append(budget_cell)
        
        return cells
    
    def _is_numeric_value(self, value: Any) -> bool:
        """Check if a value represents a numeric budget amount."""
        if isinstance(value, (int, float)):
            # Skip very small values and obvious row/column numbers
            return abs(value) >= 1 and abs(value) < 1e12
        
        if isinstance(value, str):
            # Try to parse string as number
            cleaned = re.sub(r'[,$\s]', '', value)
            try:
                num = float(cleaned)
                return abs(num) >= 1 and abs(num) < 1e12
            except ValueError:
                return False
        
        return False
    
    def _find_label_for_cell(self, sheet, row: int, col: int) -> str:
        """Find the most likely label for a numeric cell."""
        candidates = []
        
        # Check left (same row)
        for c in range(col - 1, 0, -1):
            cell_value = sheet.cell(row, c).value
            if cell_value and isinstance(cell_value, str) and cell_value.strip():
                candidates.append((cell_value.strip(), 3))  # High priority for same row
                break
        
        # Check above (same column)
        for r in range(row - 1, 0, -1):
            cell_value = sheet.cell(r, col).value
            if cell_value and isinstance(cell_value, str) and cell_value.strip():
                candidates.append((cell_value.strip(), 2))  # Medium priority
                break
        
        # Check row headers (first column)
        first_col_value = sheet.cell(row, 1).value
        if first_col_value and isinstance(first_col_value, str):
            candidates.append((first_col_value.strip(), 1))  # Lower priority
        
        # Check column headers (first row)
        first_row_value = sheet.cell(1, col).value
        if first_row_value and isinstance(first_row_value, str):
            candidates.append((first_row_value.strip(), 1))  # Lower priority
        
        # Return the highest priority candidate
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        else:
            return f"Unknown_{row}_{col}"
    
    def _find_label_for_csv_cell(self, df: pd.DataFrame, row_idx: int, col_idx: int) -> str:
        """Find label for a CSV cell."""
        # Try column header first
        if col_idx < len(df.columns):
            col_header = str(df.columns[col_idx])
            if col_header and col_header != str(col_idx):
                return col_header
        
        # Try first column of same row
        if len(df.columns) > 0:
            first_col_value = df.iloc[row_idx, 0]
            if pd.notna(first_col_value) and isinstance(first_col_value, str):
                return first_col_value
        
        return f"Unknown_{row_idx}_{col_idx}"
    
    def _get_cell_context(self, sheet, row: int, col: int, window: int = 2) -> str:
        """Get surrounding context for a cell."""
        context_parts = []
        
        # Get nearby cells
        for r in range(max(1, row - window), min(sheet.max_row + 1, row + window + 1)):
            for c in range(max(1, col - window), min(sheet.max_column + 1, col + window + 1)):
                if r == row and c == col:
                    continue  # Skip the cell itself
                
                cell_value = sheet.cell(r, c).value
                if cell_value and isinstance(cell_value, str):
                    context_parts.append(cell_value.strip())
        
        return " ".join(context_parts)
    
    def _get_csv_context(self, df: pd.DataFrame, row_idx: int, col_idx: int, window: int = 2) -> str:
        """Get surrounding context for a CSV cell."""
        context_parts = []
        
        for r in range(max(0, row_idx - window), min(len(df), row_idx + window + 1)):
            for c in range(max(0, col_idx - window), min(len(df.columns), col_idx + window + 1)):
                if r == row_idx and c == col_idx:
                    continue
                
                value = df.iloc[r, c]
                if pd.notna(value) and isinstance(value, str):
                    context_parts.append(str(value).strip())
        
        return " ".join(context_parts)
    
    def _extract_year_from_context(self, context: str) -> Optional[int]:
        """Extract year from context string."""
        if not context:
            return None
        
        # Look for year patterns
        for match in re.finditer(self.year_regex, context, re.IGNORECASE):
            year_text = match.group()
            
            # Extract just the 4-digit year
            year_match = re.search(r'20\d{2}', year_text)
            if year_match:
                return int(year_match.group())
            
            # Handle "Year N" format - assume starting from current year
            year_n_match = re.search(r'Year\s*(\d+)', year_text, re.IGNORECASE)
            if year_n_match:
                base_year = 2024  # Could be configurable
                return base_year + int(year_n_match.group(1)) - 1
        
        return None
    
    def _calculate_confidence(self, sheet, row: int, col: int) -> float:
        """Calculate confidence score for a budget cell extraction."""
        confidence = 0.5  # Base confidence
        
        # Check if we found a meaningful label
        label = self._find_label_for_cell(sheet, row, col)
        if not label.startswith("Unknown_"):
            confidence += 0.2
        
        # Check if label contains budget-related keywords
        if re.search(self.budget_regex, label, re.IGNORECASE):
            confidence += 0.2
        
        # Check if we found a year
        context = self._get_cell_context(sheet, row, col)
        if self._extract_year_from_context(context):
            confidence += 0.1
        
        return min(confidence, 1.0)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create a sample Excel file for testing
    from openpyxl import Workbook
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Budget"
    
    # Sample budget data
    ws['A1'] = "Category"
    ws['B1'] = "2024"
    ws['C1'] = "2025"
    
    ws['A2'] = "PI Salary"
    ws['B2'] = 75000
    ws['C2'] = 77250
    
    ws['A3'] = "Equipment"
    ws['B3'] = 25000
    ws['C3'] = 5000
    
    ws['A4'] = "Travel"
    ws['B4'] = 3000
    ws['C4'] = 3500
    
    test_file = Path("sample_budget.xlsx")
    wb.save(test_file)
    
    try:
        parser = BudgetParser()
        budget_book = parser.parse_file(test_file)
        
        print(f"Found {len(budget_book.cells)} budget cells:")
        for cell in budget_book.cells:
            print(f"  {cell.label}: ${cell.value:,.2f} ({cell.year}) - confidence: {cell.confidence:.2f}")
        
        # Test finding by label
        print("\nLooking for 'PI Salary':")
        matches = budget_book.find_by_label("PI_SALARY")
        for match in matches:
            print(f"  Found: ${match.value:,.2f} ({match.year})")
            
    finally:
        test_file.unlink()  # Clean up
