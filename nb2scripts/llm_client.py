"""
Azure OpenAI client for LLM-based classification.
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from openai import AzureOpenAI, BadRequestError
from .schema import Chunk, LLMClassificationResult

class LLMClient:
    """Azure OpenAI client for notebook classification."""
    
    def __init__(self, 
                 azure_endpoint: str = "https://socialstocks2.openai.azure.com/",
                 api_key: str = "",
                 api_version: str = "2024-08-01-preview",
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 cache_dir: str = ".cache"):
        
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        self.model = model
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load templates for grounding
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with better order guidance."""
        return """You are nb2scripts-LLM, an expert at analyzing Jupyter notebook content and classifying operations.

    Your job:
    1. Analyze one chunk of markdown+Python from a Jupyter notebook
    2. Classify it into exactly ONE PRIMARY operation type: 'load', 'reconcile', 'extend', or 'unknown'
    3. Extract metadata and assign SEQUENTIAL order numbers based on typical pipeline flow

    Operation Definitions and EXPECTED ORDER:

    ORDER 1 - LOAD OPERATIONS:
    - Primary purpose: Read external data and create tables
    - Key indicators: pd.read_csv(), add_table(), TableManager(), import statements
    - Should come FIRST in pipeline

    ORDER 2-3 - RECONCILE OPERATIONS:
    - Primary purpose: Match/link entities using external knowledge bases  
    - Key indicators: reconciliation_manager.reconcile(), wikidataAlligator, wikidataOpenRefine
    - Should come AFTER load operations
    - Multiple reconcile operations should get sequential orders (2,3,4...)

    ORDER 4+ - EXTEND OPERATIONS:
    - Primary purpose: Add new columns/properties to existing data
    - Key indicators: extension_manager.extend_column(), wikidataPropertySPARQL
    - Should come LAST, after reconciliation

    ORDER ASSIGNMENT RULES:
    - Load operations: order = 1
    - First reconciliation: order = 2  
    - Second reconciliation: order = 3
    - Extension operations: order = 4
    - If unsure, use order based on operation type: load=1, reconcile=2, extend=4

    IMPORTANT: Look for clues in the content about which reconciliation this is:
    - "Point of Interest" + "wikidataAlligator" = order 2
    - "Place" + "wikidataOpenRefine" = order 3

    Examples with correct orders:

    LOAD (order=1):
    ```python
    import pandas as pd
    df = pd.read_csv('./table_1.csv')
    table_id, message, table_data = table_manager.add_table(dataset_id, df, table_name)
    ```
    → {"op_type": "load", "order": 1}

    RECONCILE Point of Interest (order=2):
    ```python
    reconciliator_id = "wikidataAlligator"
    column_name = "Point of Interest"
    reconciled_table, backend_payload = reconciliation_manager.reconcile(...)
    ```
    → {"op_type": "reconcile", "order": 2}

    RECONCILE Place (order=3):
    ```python
    reconciliator_id = "wikidataOpenRefine" 
    column_name = "Place"
    reconciled_table, backend_payload = reconciliation_manager.reconcile(...)
    ```
    → {"op_type": "reconcile", "order": 3}

    EXTEND (order=4):
    ```python
    extended_table, extension_payload = extension_manager.extend_column(
        column_name="Point of Interest",
        extender_id="wikidataPropertySPARQL"
    )
    ```
    → {"op_type": "extend", "order": 4}

    Respond with valid JSON only:
    {
    "op_type": "load|reconcile|extend|unknown",
    "name": "descriptive_operation_name", 
    "order": 1,
    "meta": {
        "column_name": "extracted_column",
        "reconciliator_id": "extracted_id"
    },
    "confidence": 0.95,
    "reasoning": "Brief explanation including why this order was chosen"
    }"""

    def classify_chunk(self, chunk: Chunk) -> LLMClassificationResult:
        """
        Classify a single chunk using LLM.
        
        Args:
            chunk: Chunk to classify
            
        Returns:
            LLMClassificationResult with classification details
        """
        # Check cache first
        cache_key = self._get_cache_key(chunk)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Prepare user prompt
        user_prompt = f"""Chunk #{chunk.number} ({chunk.language}, {len(chunk.text.split())} words):

{chunk.text}

Classify this chunk and respond with JSON only."""

        try:
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Clean JSON (remove markdown formatting if present)
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            result_data = json.loads(content)
            
            # Create result object
            result = LLMClassificationResult(
                op_type=result_data.get('op_type', 'unknown'),
                name=result_data.get('name', f"unknown_{chunk.number}"),
                order=result_data.get('order', chunk.number),
                meta=result_data.get('meta', {}),
                confidence=result_data.get('confidence', 0.5),
                reasoning=result_data.get('reasoning', '')
            )
            
            # Cache the result
            self._save_to_cache(cache_key, result)
            
            return result
            
        except (BadRequestError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ LLM classification failed for chunk {chunk.number}: {e}")
            # Return unknown classification as fallback
            return LLMClassificationResult(
                op_type='unknown',
                name=f'unknown_{chunk.number}',
                order=chunk.number,
                meta={},
                confidence=0.0,
                reasoning=f'LLM classification failed: {str(e)}'
            )
    
    def classify_chunks_batch(self, chunks: List[Chunk]) -> List[LLMClassificationResult]:
        """
        Classify multiple chunks with rate limiting and error handling.
        
        Args:
            chunks: List of chunks to classify
            
        Returns:
            List of classification results
        """
        results = []
        
        print(f"🤖 Classifying {len(chunks)} chunks with LLM...")
        
        for i, chunk in enumerate(chunks):
            if i > 0 and i % 10 == 0:
                print(f"   Processed {i}/{len(chunks)} chunks")
                time.sleep(1)  # Rate limiting
            
            result = self.classify_chunk(chunk)
            results.append(result)
            
            # Print progress for high-confidence results
            if result.confidence > 0.7:
                print(f"   ✅ Chunk {chunk.number}: {result.op_type} ({result.confidence:.2f})")
            else:
                print(f"   ⚠️ Chunk {chunk.number}: {result.op_type} ({result.confidence:.2f}) - {result.reasoning}")
        
        return results
    
    def _get_cache_key(self, chunk: Chunk) -> str:
        """Generate cache key for chunk."""
        content_hash = hashlib.md5(chunk.text.encode()).hexdigest()
        return f"{self.model}_{content_hash}_{chunk.language}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[LLMClassificationResult]:
        """Load result from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return LLMClassificationResult(**data)
            except (json.JSONDecodeError, TypeError):
                # Invalid cache file, ignore
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, result: LLMClassificationResult):
        """Save result to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'op_type': result.op_type,
                    'name': result.name,
                    'order': result.order,
                    'meta': result.meta,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to cache result: {e}")