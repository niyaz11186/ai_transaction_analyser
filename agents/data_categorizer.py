"""Data Categorizer - Categorizes transactions based on cleaned remarks."""

from typing import Dict
import pandas as pd
import json
import re


class DataCategorizer:
    """
    Categorizer agent for classifying transactions.
    
    Uses LLM to categorize transactions based on cleaned transaction remarks
    dynamically, allowing for flexible category discovery while maintaining consistency.
    """
    
    SYSTEM_PROMPT = """You are an intelligent financial categorisation agent.

Goal:
Given raw UPI/bank statement transactions, analyse and categorise each transaction based on its purpose or intent.

You must infer meaningful, human-understandable categories, not limited to any predefined list.

Input format:
Each transaction contains:
- S No.
- Transaction Date
- Transaction Remarks
- Withdrawal Amount(INR)
- Deposit Amount(INR)  
- Balance(INR)

For each transaction:

1. Understand the intent from the Transaction Remarks field — who, what, why.

2. Assign a short, meaningful category name that reflects the spending type or transaction purpose.
   - You may create new category names dynamically if they fit the context better.
   - Example categories: "Groceries", "Fuel", "Bills", "Transfers", "Work Credit", "Refunds", 
     "Online Shopping", "Family Support", "Subscriptions", "Medical", "Cash Withdrawals", 
     "Food & Beverage", "Travel", "Utilities", "Savings", etc.
   - Be concise, but descriptive (1–3 words).
   - Use consistent naming for similar transactions (e.g., always use "Fuel" not "Petrol" or "Gas")

3. Provide a confidence score (High / Medium / Low) for how sure you are of the classification.

Guidelines for intelligent inference:
- Use context clues like vendor names, UPI handles, keywords (Amazon, Airtel, IRCTC, Paytm, etc.), and transaction amounts.
- Infer semantics rather than relying on strict keyword matching.
- For unclear or vague remarks, infer probable intent (e.g., vendor name + small amount likely = Grocery or Food).
- If truly ambiguous, mark category as "Unclear" and Confidence = Low.
- For income/credits: distinguish between Salary, Family Support, Work Credit, Refunds, etc.
- For transfers: distinguish between Savings transfers, Internal transfers, etc.
- For expenses: be specific (e.g., "Groceries" vs "Food & Beverage" vs "Fuel")

Output Format:
You must respond with a JSON object containing exactly two fields:
{
    "category": "Category name (1-3 words, concise and descriptive)",
    "confidence": "High" or "Medium" or "Low"
}

Examples:
Input: Transaction Remarks="Petrol purchase for car", Withdrawal Amount(INR)=1000, Deposit Amount(INR)=0
Output: {"category": "Fuel", "confidence": "High"}

Input: Transaction Remarks="Temporary reversal (refund back to Kotak Mahindra account)", Withdrawal Amount(INR)=0, Deposit Amount(INR)=800
Output: {"category": "Refund", "confidence": "High"}

Input: Transaction Remarks="Savings October transfer to Kotak Mahindra account", Withdrawal Amount(INR)=16500, Deposit Amount(INR)=0
Output: {"category": "Savings Transfer", "confidence": "High"}

Input: Transaction Remarks="UPI payment to unknown vendor", Withdrawal Amount(INR)=250, Deposit Amount(INR)=0
Output: {"category": "Unclear", "confidence": "Low"}

Remember: Always respond with valid JSON only, no additional text before or after."""

    def __init__(self, llm_client):
        """
        Initialize the Data Categorizer.
        
        Args:
            llm_client: LLM client instance for categorization
        """
        self.llm_client = llm_client
    
    def categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize transactions based on cleaned remarks.
        
        Args:
            df: DataFrame with transaction data (should have 'Cleaned Remark' column)
            
        Returns:
            DataFrame with added 'Category' and 'Confidence' columns
        """
        # Determine which description column to use
        if 'Cleaned Remark' in df.columns:
            desc_column = 'Cleaned Remark'
        elif 'Transaction Remarks' in df.columns:
            desc_column = 'Transaction Remarks'
        else:
            raise ValueError("DataFrame must contain 'Cleaned Remark' or 'Transaction Remarks' column")
        
        categories = []
        confidences = []
        
        total_rows = len(df)
        print(f"Categorizing {total_rows} transactions...")
        
        for idx, row in df.iterrows():
            description = str(row[desc_column]) if pd.notna(row[desc_column]) else ""
            withdrawal = float(row.get('Withdrawal Amount(INR)', 0) or 0)
            deposit = float(row.get('Deposit Amount(INR)', 0) or 0)
            
            if not description.strip():
                categories.append("Unclear")
                confidences.append("Low")
                continue
            
            try:
                result = self.categorize_single_transaction(description, withdrawal, deposit)
                categories.append(result['category'])
                confidences.append(result['confidence'])
                
                # Progress indicator
                if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
                    print(f"  Categorized {idx + 1}/{total_rows} transactions...", end='\r')
                    
            except Exception as e:
                print(f"\n  Warning: Error categorizing transaction at row {idx + 1}: {str(e)}")
                categories.append("Unclear")
                confidences.append("Low")
        
        print()  # New line after progress
        
        # Add columns
        df['Category'] = categories
        df['Confidence'] = confidences
        
        return df
    
    def categorize_single_transaction(
        self, 
        cleaned_remark: str, 
        withdrawal: float = 0.0, 
        deposit: float = 0.0
    ) -> Dict[str, str]:
        """
        Categorize a single transaction.
        
        Args:
            cleaned_remark: Cleaned transaction remark
            withdrawal: Withdrawal amount (INR)
            deposit: Deposit amount (INR)
            
        Returns:
            Dictionary with 'category' and 'confidence' keys
        """
        user_prompt = f"""Analyze and categorize this transaction:

Transaction Remarks: {cleaned_remark}
Withdrawal Amount(INR): ₹{withdrawal:,.2f}
Deposit Amount(INR): ₹{deposit:,.2f}

Respond with JSON only: {{"category": "...", "confidence": "..."}}"""
        
        response = None
        try:
            response = self.llm_client.invoke(user_prompt, self.SYSTEM_PROMPT)
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*"category"[^{}]*"confidence"[^{}]*\}', response, re.DOTALL)
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
            category = result.get('category', 'Unclear').strip()
            confidence = result.get('confidence', 'Low').strip()
            
            # Validate confidence value
            if confidence not in ['High', 'Medium', 'Low']:
                confidence = 'Low'
            
            # Validate category is not empty
            if not category:
                category = 'Unclear'
            
            return {
                'category': category,
                'confidence': confidence
            }
            
        except json.JSONDecodeError as e:
            # Fallback: try to extract meaningful parts
            fallback_category = "Unclear"
            if response:
                # Try to infer category from response text
                response_lower = response.lower()
                if any(word in response_lower for word in ['fuel', 'petrol', 'diesel']):
                    fallback_category = "Fuel"
                elif any(word in response_lower for word in ['grocery', 'food', 'restaurant']):
                    fallback_category = "Food"
                elif any(word in response_lower for word in ['refund', 'reversal']):
                    fallback_category = "Refund"
            
            return {
                'category': fallback_category,
                'confidence': 'Low'
            }
        except Exception as e:
            return {
                'category': 'Unclear',
                'confidence': 'Low'
            }

