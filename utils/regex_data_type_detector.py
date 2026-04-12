import re
import pandas as pd

class RegexDataTypeDetector:
    """Detects data types based on pattern matching"""

    def __init__(self):
        # Date separators
        self.date_separators = ['-', '/', '.', ' ']

        # Date patterns: (YY)YYx(M)Mx(D)D, (M)Mx(D)Dx(YY)YY, (D)Dx(M)Mx(YY)YY
        self.date_patterns = [
            # YYYY-MM-DD or YY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
            r'^\d{2,4}[/.\s-]\d{1,2}[/.\s-]\d{1,2}$',
            # MM-DD-YYYY or MM-DD-YY or MM/DD/YYYY
            r'^\d{1,2}[/.\s-]\d{1,2}[/.\s-]\d{2,4}$',
            # DD-MM-YYYY or DD-MM-YY or DD/MM/YYYY
            r'^\d{1,2}[/.\s-]\d{1,2}[/.\s-]\d{2,4}$',
        ]

        # Time patterns: HH:MM:SS, HH:MM, H:MM
        self.time_patterns = [
            r'^\d{1,2}:\d{2}:\d{2}$',  # HH:MM:SS
            r'^\d{1,2}:\d{2}$',  # HH:MM or H:MM
        ]

        # Timestamp patterns: date + time (+ optional timezone)
        self.timestamp_patterns = [
            # YYYY-MM-DD HH:MM:SS or YYYY-MM-DDTHH:MM:SS
            r'^\d{2,4}[/.\s-]\d{1,2}[/.\s-]\d{1,2}[\sT]\d{1,2}:\d{2}(:\d{2})?',
            # MM-DD-YYYY HH:MM:SS or MM/DD/YYYY HH:MM
            r'^\d{1,2}[/.\s-]\d{1,2}[/.\s-]\d{2,4}[\sT]\d{1,2}:\d{2}(:\d{2})?',
            # With timezone: YYYY-MM-DD HH:MM:SS+00:00 or Z
            r'^\d{2,4}[/.\s-]\d{1,2}[/.\s-]\d{1,2}[\sT]\d{1,2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)?',
        ]

    def is_boolean(self, values):
        """Check if all values are boolean (0/1 or True/False)"""
        unique_values = set(str(v).strip().lower() for v in values if pd.notna(v))

        return unique_values == {'0', '1'} or unique_values.issubset({'true', 'false'})

    def is_date(self, value):
        """Check if value matches date pattern"""
        if pd.isna(value):
            return False

        value_str = str(value).strip()

        for pattern in self.date_patterns:
            if re.match(pattern, value_str):
                return True

        return False

    def is_time(self, value):
        """Check if value matches time pattern"""
        if pd.isna(value):
            return False

        value_str = str(value).strip()

        for pattern in self.time_patterns:
            if re.match(pattern, value_str):
                return True

        return False

    def is_timestamp(self, value):
        """Check if value matches timestamp pattern"""
        if pd.isna(value):
            return False

        value_str = str(value).strip()

        for pattern in self.timestamp_patterns:
            if re.match(pattern, value_str):
                return True

        return False

    def is_numeric(self, values):
        """Check if all values are numeric"""
        try:
            for v in values:
                if pd.notna(v):
                    # Remove commas for numbers like 1,000
                    float(str(v).replace(',', '').strip())
            return True
        except (ValueError, TypeError):
            return False

    def is_varchar(self, value):
        """Check if value is varchar (text/alphanumeric with letters or special chars)"""
        if pd.isna(value):
            return False

        value_str = str(value).strip()

        if not value_str:  # Empty string
            return False

        # Varchar if it contains:
        # 1. Any letter (a-z, A-Z)
        # 2. Any special character (not digit, not whitespace)
        has_letters = bool(re.search(r'[a-zA-Z]', value_str))
        has_special_chars = bool(re.search(r'[^0-9]', value_str))

        return has_letters or has_special_chars

    def detect_column_type(self, column_data):
        valid_data = column_data.dropna()
        valid_data = valid_data[valid_data.astype(str).str.strip(" '\"") != '']
        if len(valid_data) == 0:
            return 'no_data'

        values = valid_data.tolist()

        if self.is_boolean(values):
            return 'boolean'

        if self.is_numeric(values):
            return 'number'

        # Single pass: classify each value once
        counts = {'timestamp': 0, 'date': 0, 'time': 0, 'varchar': 0}
        n = len(values)
        for v in values:
            if self.is_timestamp(v):
                counts['timestamp'] += 1
            elif self.is_date(v):
                counts['date'] += 1
            elif self.is_time(v):
                counts['time'] += 1
            elif self.is_varchar(v):
                counts['varchar'] += 1

        for dtype in ('timestamp', 'date', 'time'):
            if counts[dtype] == n:
                return dtype
        if counts['varchar'] > 0:
            return 'varchar'
        return 'unknown'
