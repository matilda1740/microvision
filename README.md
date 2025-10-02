# MicroVision - Microservice Dependency Mapping

A semantic log analysis system for automated microservice dependency discovery.

## Quick Start

1. Activate environment: `source venv/bin/activate`
2. Test installation: `python3 test_drain3.py`
3. Launch dashboard: `streamlit run src/visualization/streamlit_app.py`

## Project Structure
- `src/ingestion/` - Log collection and processing
- `src/parsing/` - Semantic log parsing with Drain3
- `src/storage/` - ChromaDB for template storage
- `src/visualization/` - Streamlit dashboard
