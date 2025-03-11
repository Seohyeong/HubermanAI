# run the server: 
cd api
uvicorn main:app --reload

# run the frontend ui
cd ../app
streamlit run app.py

# mlflow
mlflow server --host 127.0.0.1 --port 8080 # terminal (before running experiments)
# http://localhost:8080/