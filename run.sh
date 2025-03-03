# run the server: 
cd api
uvicorn main:app --reload

# run the frontend ui
cd ../app
streamlit run app.py