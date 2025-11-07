"""Main entry point for the Bank Statement AI Analyser."""

import sys
import argparse
from pathlib import Path

from utils.file_handler import load_excel, save_csv, get_summary_stats, print_summary_stats
from utils.llm_client import LLMClient
from agents.transaction_remark_expert import TransactionRemarkExpert
from agents.data_categorizer import DataCategorizer
from chat.interface import ChatInterface
from config.settings import Settings


def process_transactions(input_file: str, output_file: str) -> None:
    """
    Process transactions: normalize remarks and categorize.
    
    Args:
        input_file: Path to input Excel file
        output_file: Path to output CSV file
    """
    print(f"Loading Excel file: {input_file}")
    df = load_excel(input_file)
    print(f"Loaded {len(df)} transactions\n")
    
    # Initialize LLM client
    print("Initializing LLM client...")
    llm_client = LLMClient()
    print(f"Connected to Ollama model: {Settings.get_model_name()}\n")
    
    # Process with Transaction Remark Expert
    print("Processing transaction remarks...")
    remark_expert = TransactionRemarkExpert(llm_client)
    df = remark_expert.process_remarks(df)
    print("Transaction remarks processed.\n")
    
    # Process with Data Categorizer
    print("Categorizing transactions...")
    categorizer = DataCategorizer(llm_client)
    df = categorizer.categorize_transactions(df)
    print("Transactions categorized.\n")
    
    # Save processed data
    print(f"Saving processed data to: {output_file}")
    save_csv(df, output_file)
    
    # Print summary statistics
    stats = get_summary_stats(df)
    print_summary_stats(stats)
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bank Statement AI Analyser - Process transactions and chat about your spending"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input Excel file (.xlsx)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_file name with .csv extension)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.xlsx':
        print(f"Error: Input file must be an Excel file (.xlsx)")
        sys.exit(1)
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        output_file = str(input_path.with_suffix('.csv'))
    
    try:
        # Process transactions
        df = process_transactions(str(input_path), output_file)
        
        # Initialize chat interface
        print("\n" + "="*50)
        print("Starting Chat Interface...")
        print("="*50)
        
        llm_client = LLMClient()
        chat = ChatInterface(df, llm_client)
        chat.start_chat()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

