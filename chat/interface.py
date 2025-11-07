"""CLI chat interface for querying transaction data."""

import pandas as pd
from typing import Optional
from utils.llm_client import LLMClient


class ChatInterface:
    """
    Chat interface for conversational queries about transaction data.
    
    Allows users to ask questions like:
    - "How much spending did I do on food?"
    - "How can I save on food?"
    - "What are my top spending categories?"
    """
    
    def __init__(self, df: pd.DataFrame, llm_client: LLMClient):
        """
        Initialize the chat interface.
        
        Args:
            df: Processed DataFrame with transaction data
            llm_client: LLM client instance
        """
        self.df = df
        self.llm_client = llm_client
    
    def start_chat(self) -> None:
        """
        Start the interactive chat loop.
        """
        print("\n" + "="*50)
        print("CHAT INTERFACE")
        print("Ask questions about your transactions!")
        print("Type 'exit' or 'quit' to end the chat.")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = self.process_query(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return response.
        
        Args:
            query: User's question/query
            
        Returns:
            LLM-generated response
        """
        # TODO: Implement query processing
        # This will:
        # 1. Prepare context from DataFrame (summary stats, category breakdowns, etc.)
        # 2. Create a system prompt with transaction data context
        # 3. Send query to LLM with context
        # 4. Return formatted response
        
        # Placeholder implementation
        context = self._prepare_context()
        system_prompt = self._create_system_prompt(context)
        
        response = self.llm_client.invoke(query, system_prompt)
        return response
    
    def _prepare_context(self) -> dict:
        """
        Prepare context data from DataFrame for LLM.
        
        Returns:
            Dictionary with relevant context information
        """
        context = {
            'total_transactions': len(self.df),
            'date_range': None,
            'categories': {}
        }
        
        if 'Transaction Date' in self.df.columns:
            dates = pd.to_datetime(self.df['Transaction Date'], errors='coerce',dayfirst=True)
            context['date_range'] = {
                'start': dates.min().strftime('%Y-%m-%d') if not dates.isna().all() else None,
                'end': dates.max().strftime('%Y-%m-%d') if not dates.isna().all() else None
            }
        
        # Find Category column (case-insensitive)
        category_col = None
        for col in self.df.columns:
            if col.lower() == 'category':
                category_col = col
                break
        
        if category_col and 'Withdrawal Amount(INR)' in self.df.columns:
            category_spending = self.df.groupby(category_col)['Withdrawal Amount(INR)'].sum().to_dict()
            context['categories'] = category_spending
        
        return context
    
    def _create_system_prompt(self, context: dict) -> str:
        """
        Create system prompt with transaction data context.
        
        Args:
            context: Context dictionary from _prepare_context
            
        Returns:
            System prompt string
        """
        prompt = f"""You are a helpful financial assistant analyzing bank transaction data.

Transaction Summary:
- Total Transactions: {context['total_transactions']}
"""
        
        if context['date_range'] and context['date_range']['start']:
            prompt += f"- Date Range: {context['date_range']['start']} to {context['date_range']['end']}\n"
        
        if context['categories']:
            prompt += "\nCategory-wise Spending:\n"
            for category, amount in context['categories'].items():
                prompt += f"- {category}: ₹{amount:,.2f}\n"
        
        prompt += """
Answer questions about spending patterns, provide insights, and suggest ways to save money.
Be concise and helpful.
"""
        
        return prompt

