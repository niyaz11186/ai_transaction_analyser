"""Transaction Remark Expert - Normalizes and cleans transaction remarks using LLM."""

from typing import Dict, List
import pandas as pd
import json
import re


class TransactionRemarkExpert:
    """
    Expert agent for normalizing transaction remarks.
    
    Uses LLM to interpret abbreviated, shorthand, or truncated UPI/NEFT/IMPS
    payment references and generate cleaned remarks with notes/doubts.
    """
    
    SYSTEM_PROMPT = """You are a transaction remark interpretation specialist.

Your primary focus is on the column "Transaction Remarks", which contains UPI / NEFT / IMPS payment references, often written in abbreviated, shorthand, or truncated form.

Your job is to read, interpret, and normalize these remarks using contextual and linguistic patterns.

Core Requirements:
- Focus exclusively on the "Transaction Remarks" text to derive meaning.
- Ignore IDs, numbers, and bank codes (e.g., 563842971184, ICIf635d23e7a...).
- Look for meaningful tokens such as:
  * Names (e.g., SHAIK NIYA, SADIYA BEG, APOLLO PHA)
  * Purpose words (e.g., rent, milk, grocery, temp rever, ticket, savings oc)
  * Abbreviations (e.g., oc = October, au = August, rever = reversal, bik = bike)
  * Platforms (e.g., Google Pay, Paytm, BharatPe, Amazon Pay, IRCTC)
  * Contextual verbs (e.g., Payment fr, Pay to, refund, transfer)
- Extract the real-life meaning behind the transaction — e.g., what was paid for or received.

Inference Logic:
- Pattern-based: Detect recurring patterns (temp rever → temporary reversal, savings oc → savings October, etc.)
- Frequency-based: If multiple similar rows exist, assume consistent context
- Semantic mapping: Translate partial abbreviations (bik → bike, rever → reversal, oc → October)
- Priority: purpose > person > platform > reference number

Output Format:
You must respond with a JSON object containing exactly two fields:
{
    "cleaned_remark": "A short, natural-language interpretation of what the transaction likely represents",
    "notes_doubts": "Short reasoning only if uncertain, otherwise use '—' or empty string"
}

Examples:
Input: "UPI/SHAIK NIYA/niyazahamed5@o/temp rever/Kotak Mahi/563842971184/ICIf635d23e7afb45db8a617a6b9ea0020c"
Output: {"cleaned_remark": "Temporary reversal (refund back to Kotak Mahindra account)", "notes_doubts": "—"}

Input: "UPI/SHAIK NIYA/niyazahamed5@o/savings oc/Kotak Mahi/563839565384/ICId2597140615b4e8daf995d05f0a9690a"
Output: {"cleaned_remark": "Savings October transfer to Kotak Mahindra account", "notes_doubts": "—"}

Remember: Always respond with valid JSON only, no additional text before or after."""

    def __init__(self, llm_client):
        """
        Initialize the Transaction Remark Expert.
        
        Args:
            llm_client: LLM client instance for processing remarks
        """
        self.llm_client = llm_client
    
    def process_remarks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process transaction remarks and add cleaned columns.
        
        Args:
            df: DataFrame with 'Transaction Remarks' column
            
        Returns:
            DataFrame with added 'Cleaned Remark' and 'Notes / Doubts' columns
        """
        if 'Transaction Remarks' not in df.columns:
            raise ValueError("DataFrame must contain 'Transaction Remarks' column")
        
        cleaned_remarks = []
        notes_doubts = []
        
        total_rows = len(df)
        print(f"Processing {total_rows} transaction remarks...")
        
        for idx, row in df.iterrows():
            remark = str(row['Transaction Remarks']) if pd.notna(row['Transaction Remarks']) else ""
            
            if not remark.strip():
                cleaned_remarks.append("")
                notes_doubts.append("")
                continue
            
            try:
                result = self.normalize_single_remark(remark)
                cleaned_remarks.append(result['cleaned_remark'])
                notes_doubts.append(result['notes_doubts'])
                
                # Progress indicator
                if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
                    print(f"  Processed {idx + 1}/{total_rows} remarks...", end='\r')
                    
            except Exception as e:
                print(f"\n  Warning: Error processing remark at row {idx + 1}: {str(e)}")
                cleaned_remarks.append("")
                notes_doubts.append(f"Error: {str(e)}")
        
        print()  # New line after progress
        
        # Add columns with exact names as specified
        df['Cleaned Remark'] = cleaned_remarks
        df['Notes / Doubts'] = notes_doubts
        
        return df
    
    def normalize_single_remark(self, remark: str) -> Dict[str, str]:
        """
        Normalize a single transaction remark.
        
        Args:
            remark: Raw transaction remark string
            
        Returns:
            Dictionary with 'cleaned_remark' and 'notes_doubts' keys
        """
        user_prompt = f"""Analyze this transaction remark and provide cleaned interpretation:

Transaction Remark: {remark}

Respond with JSON only: {{"cleaned_remark": "...", "notes_doubts": "..."}}"""
        
        response = None
        try:
            response = self.llm_client.invoke(user_prompt, self.SYSTEM_PROMPT)
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{[^{}]*"cleaned_remark"[^{}]*"notes_doubts"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Validate and extract fields
            cleaned_remark = result.get('cleaned_remark', '').strip()
            notes_doubts = result.get('notes_doubts', '').strip()
            
            # Replace empty notes with dash
            if not notes_doubts or notes_doubts.lower() in ['—', '-', 'none', 'null']:
                notes_doubts = '—'
            
            return {
                'cleaned_remark': cleaned_remark,
                'notes_doubts': notes_doubts
            }
            
        except json.JSONDecodeError as e:
            # Fallback: try to extract meaningful parts
            fallback_text = response[:200] if response else "Unable to parse"
            return {
                'cleaned_remark': fallback_text,
                'notes_doubts': f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            return {
                'cleaned_remark': '',
                'notes_doubts': f"Error: {str(e)}"
            }

