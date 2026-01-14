cd logistics_streamlit_starter/infra
docker compose up -d
cd ../
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app_streamlit/app.py
