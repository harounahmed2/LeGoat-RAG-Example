# LeBron's Legacy Tracker

## Overview
LeBron's Legacy Tracker leverages the Retrieval Augmented Generation (RAG) technique alongside static documentation to provide updated and focused responses on LeBron James' career achievements. This project demonstrates how RAG can be used to enhance Large Language Models' (LLMs) understanding by directing attention to the most relevant information, specifically LeBron James' statistics and historical achievements.

## Key Features
- **RAG Implementation**: Integrates RAG to augment LLM's responses with specific, up-to-date information from static sources.
- **Static Information Utilization**: Uses static documentation, such as HTML pages from Basketball Reference, to provide LLMs with a focused knowledge base.
- **Streamlit Interface**: Offers an interactive web interface for users to query LeBron James' career stats and milestones.

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- virtualenv

### Installation

1. **Create and activate a virtual environment**
```
    python -m venv venv
    source venv/bin/activate
```

2. **Install requirements**
```
pip install -r requirements.txt
```
3. **Set up Streamlit secrets**
To avoid deploying secrets to GitHub, add `.streamlit` to your `.gitignore` file:
```
echo '.streamlit/' >> .gitignore
```
Then, create a `secrets.toml` file in `.streamlit` directory locally to store your API keys and other secrets:
```
mkdir -p .streamlit
echo 'openai_key="YOUR_OPENAI_API_KEY"' > .streamlit/secrets.toml
```

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

### Running the Application

To start the Streamlit app, run:
```
streamlit run app.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
