from functools import lru_cache
from typing import Any, Type, Dict, List, Tuple
import pandas as pd
from pydantic import BaseModel, ValidationError

from politipo.core.conversion.strategies.base import ConversionStrategy
from politipo.core.types import CanonicalType
from politipo.core.conversion.engine import ConversionContext
from politipo.core.errors import ConversionError


class DataFrameToModelStrategy(ConversionStrategy):
    """Strategy for converting Pandas DataFrames to Pydantic models."""

    def can_handle(self, source: CanonicalType, target: CanonicalType) -> bool:
        """Check if this strategy can handle the conversion."""
        return (
            source.kind == "composite" and source.name == "DataFrame"
            and target.kind == "composite" and target.name == "Model"
        )

    def convert(self, df: pd.DataFrame, context: ConversionContext) -> List[BaseModel]:
        """
        Convert DataFrame rows to Pydantic model instances.
        
        Args:
            df: Source DataFrame
            context: Conversion context
            
        Returns:
            List of Pydantic model instances
            
        Raises:
            ConversionError: If conversion fails and strict mode is enabled
        """
        try:
            model_type = context.target_type_system.from_canonical(context.target)
            results = []
            errors = []
            
            for idx, row in df.iterrows():
                try:
                    data = {
                        k: None if pd.isna(v) else v 
                        for k, v in row.items()
                        if k in model_type.model_fields
                    }
                    results.append(
                        self._convert_row(
                            self._make_row_hashable(data),
                            model_type
                        )
                    )
                except ValidationError as e:
                    if context.strict:
                        raise ConversionError(
                            f"Row {idx} validation failed: {str(e)}"
                        ) from e
                    errors.append((idx, str(e)))
                    
            if errors and not context.strict:
                print(
                    f"Warning: {len(errors)} rows failed validation:\n" +
                    "\n".join(f"Row {idx}: {msg}" for idx, msg in errors)
                )
                
            return results
            
        except Exception as e:
            raise ConversionError(
                f"Failed to convert DataFrame to {model_type.__name__}"
            ) from e

    @lru_cache(maxsize=1024)
    def _convert_row(
        self,
        row_hash: Tuple[Tuple[str, Any], ...],
        model_type: Type[BaseModel]
    ) -> BaseModel:
        """
        Optimized row conversion with caching.
        
        Args:
            row_hash: Hashable representation of row data
            model_type: Target Pydantic model class
            
        Returns:
            Model instance
            
        Raises:
            ValidationError: If row data is invalid
        """
        return model_type(**dict(row_hash))

    def _make_row_hashable(self, row_dict: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        """
        Convert row dictionary to hashable format for caching.
        
        Args:
            row_dict: Row data dictionary
            
        Returns:
            Hashable representation of row data
        """
        return tuple(sorted(
            (k, v) if isinstance(v, (str, int, float, bool, type(None)))
            else (k, str(v))
            for k, v in row_dict.items()
        )) 