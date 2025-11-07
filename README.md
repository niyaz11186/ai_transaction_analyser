# AI Transaction Analyser

An intelligent bank statement analyser that uses AI to normalize transaction remarks, categorize expenses, and provide conversational insights about your spending patterns.

## Features

- **Transaction Remark Normalization**: Automatically interprets and cleans abbreviated UPI/NEFT/IMPS transaction remarks using AI
- **Intelligent Categorization**: Dynamically categorizes transactions into meaningful categories (e.g., Groceries, Fuel, Bills, Transfers)
- **Interactive Chat Interface**: Ask questions about your spending patterns and get AI-powered insights
- **Summary Statistics**: Get comprehensive summaries of your transaction data including category breakdowns

## Prerequisites

- Python 3.12 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- An Ollama model (default: `gemma3:latest`, configurable)

### Installing Ollama

1. Visit [https://ollama.ai/](https://ollama.ai/) and install Ollama for your platform
2. Pull a model (e.g., `gemma3:latest`):
   ```bash
   ollama pull gemma3:latest
   ```
3. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai_transaction_analyser
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The application uses environment variables for configuration. You can set them in a `.env` file or export them:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434  # Default: http://localhost:11434
OLLAMA_MODEL=gemma3:latest               # Default: gemma3:latest

# Processing Configuration
OUTPUT_DIR=./output                      # Default: ./output

# LLM Configuration
TEMPERATURE=0.1                         # Default: 0.1
```

### Example `.env` file

Create a `.env` file in the project root:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:latest
TEMPERATURE=0.1
```

## Usage

### Basic Usage

Process a bank statement Excel file:

```bash
python main.py path/to/your/statement.xlsx
```

The processed data will be saved as a CSV file with the same name (e.g., `statement.csv`).

### Specify Output File

```bash
python main.py path/to/your/statement.xlsx -o output.csv
```

### Expected Input Format

Your Excel file should contain the following columns:
- `S No.`
- `Transaction Date`
- `Transaction Remarks`
- `Withdrawal Amount(INR)`
- `Deposit Amount(INR)`
- `Balance(INR)`

### Output

The processed CSV file will include:
- Original columns
- `Cleaned Remark`: Normalized interpretation of transaction remarks
- `Notes / Doubts`: Any uncertainties or notes about the interpretation
- `Category`: AI-assigned category for each transaction
- `Confidence`: Confidence level (High/Medium/Low) for the categorization

### Interactive Chat

After processing, an interactive chat interface will start automatically. You can ask questions like:
- "How much did I spend on food?"
- "What are my top spending categories?"
- "How can I save on groceries?"
- "Show me my spending trends"

Type `exit` or `quit` to end the chat session.

## Project Structure

```
ai_transaction_analyser/
├── agents/
│   ├── data_categorizer.py          # Transaction categorization agent
│   └── transaction_remark_expert.py  # Transaction remark normalization agent
├── chat/
│   └── interface.py                  # Interactive chat interface
├── config/
│   └── settings.py                   # Configuration settings
├── utils/
│   ├── file_handler.py               # Excel/CSV file operations
│   └── llm_client.py                 # LLM client wrapper (LangChain + Ollama)
├── main.py                           # Main entry point
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## How It Works

1. **Transaction Remark Processing**: The `TransactionRemarkExpert` agent uses AI to interpret abbreviated transaction remarks, extracting meaningful information from UPI/NEFT/IMPS references.

2. **Categorization**: The `DataCategorizer` agent analyzes cleaned remarks and transaction amounts to assign categories dynamically, allowing for flexible category discovery.

3. **Chat Interface**: The chat interface provides context-aware responses about your spending patterns, using summary statistics and category breakdowns.

## Dependencies

- `langchain` - LLM framework
- `langchain-community` - Community integrations
- `langchain-core` - Core LangChain functionality
- `pandas` - Data manipulation and analysis
- `openpyxl` - Excel file support
- `python-dotenv` - Environment variable management

## Troubleshooting

### Ollama Connection Issues

If you encounter connection errors:
1. Ensure Ollama is running: `ollama serve`
2. Verify the model is available: `ollama list`
3. Check `OLLAMA_BASE_URL` in your configuration

### Model Not Found

If the specified model is not available:
1. Pull the model: `ollama pull gemma3:latest`
2. Or change `OLLAMA_MODEL` in your configuration to an available model

### Processing Errors

- Ensure your Excel file has all required columns
- Check that transaction dates are in a valid format
- Verify that amount columns contain numeric values

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

[https://github.com/niyaz11186  niyazahamed5@gmail.com]

