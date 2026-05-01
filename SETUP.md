# Quick Setup Guide 🚀

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Ensure Data Exists

If you don't have `data/customer_churn.csv`, generate it:
```bash
python data/dataset_generation.py
```

## Step 3: Train the Model (if not already trained)

```bash
python model_training.py
```

## Step 4: Run the Application

### Windows:
```bash
run_app.bat
```
OR
```bash
streamlit run app/streamlit_app.py
```

### Linux/Mac:
```bash
chmod +x run_app.sh
./run_app.sh
```
OR
```bash
streamlit run app/streamlit_app.py
```

## Step 5: Access the Application

Open your browser and go to:
```
http://localhost:8501
```

## Troubleshooting

### Issue: Model files not found
**Solution**: Run `python model_training.py` first

### Issue: Module not found
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: Port already in use
**Solution**: Change the port in `.streamlit/config.toml` or use:
```bash
streamlit run app/streamlit_app.py --server.port=8502
```

## Need Help?

Check the main README.md for detailed documentation.
