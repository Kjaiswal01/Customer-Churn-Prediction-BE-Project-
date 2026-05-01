# 🚀 Customer Churn Prediction Platform - Flask Edition

## ✨ Enhanced Features

### 🎨 Modern Professional UI
- **Beautiful gradient theme** with professional teal/blue color scheme
- **Responsive design** that works on all devices
- **Smooth animations** with AOS (Animate On Scroll)
- **Interactive charts** using Chart.js
- **Modern Bootstrap 5** components

### 🔮 Advanced Features

#### 1. **Customer Health Scoring**
   - Real-time health score calculation (0-100)
   - Health categories: Excellent, Good, Fair, Poor
   - Multi-factor health assessment

#### 2. **Predictive CLV (Customer Lifetime Value)**
   - Calculate expected customer lifetime value
   - Risk-adjusted CLV predictions
   - Revenue optimization insights

#### 3. **Comprehensive Risk Assessment**
   - Multi-factor risk scoring
   - Risk categories: Low, Medium, High
   - Visual risk indicators

#### 4. **Smart Recommendations**
   - AI-generated actionable recommendations
   - Priority-based action items
   - Personalized retention strategies

#### 5. **Customer Segmentation**
   - Automatic customer categorization
   - Risk-based segmentation
   - Segment-specific strategies

#### 6. **Campaign Management**
   - Create retention campaigns
   - Target specific customer segments
   - Campaign templates and tools

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

**Windows:**
```bash
run_flask.bat
```

**Linux/Mac:**
```bash
chmod +x run_flask.sh
./run_flask.sh
```

**Or manually:**
```bash
python app.py
```

### Step 3: Access the Application
Open your browser and go to:
```
http://localhost:5000
```

## 📁 Project Structure

```
customer-churn-prediction/
├── app.py                 # Flask application (main)
├── templates/             # HTML templates
│   ├── base.html         # Base template
│   ├── dashboard.html    # Dashboard page
│   ├── predict.html      # Prediction page
│   ├── analytics.html    # Analytics page
│   ├── customers.html    # Customer insights
│   ├── retention.html    # Retention strategies
│   └── campaigns.html    # Campaign management
├── static/               # Static files
│   ├── css/
│   │   └── style.css    # Custom styles
│   ├── js/
│   │   └── main.js      # JavaScript utilities
│   └── images/          # Images
├── models/              # ML models
├── data/               # Data files
└── requirements.txt    # Python dependencies
```

## 🎯 Key Features

### Dashboard
- Real-time metrics and statistics
- Interactive charts and visualizations
- Quick actions and navigation
- Key feature highlights

### Predict Churn
- Single customer prediction
- Risk and health scoring
- CLV calculation
- Sentiment analysis
- Actionable recommendations
- Visual risk gauges

### Analytics
- Comprehensive customer analytics
- Churn by subscription type
- Churn by gender
- Trend analysis
- Interactive charts

### Customer Insights
- Bulk customer analysis
- Customer segmentation
- Risk categorization
- Health scoring
- Export capabilities

### Retention Strategies
- Retention framework by risk level
- Email templates
- Action plans
- Best practices

### Campaign Management
- Create retention campaigns
- Target customer segments
- Campaign templates
- Campaign tracking (coming soon)

## 🎨 Color Scheme

The platform uses a professional color scheme:
- **Primary**: Blue gradient (#667eea to #764ba2)
- **Success**: Green (#198754)
- **Warning**: Yellow/Orange (#ffc107)
- **Danger**: Red (#dc3545)
- **Info**: Teal/Blue (#0dcaf0)

## 🔧 API Endpoints

### Prediction API
- `POST /api/predict` - Predict churn for a single customer

### Analytics API
- `GET /api/dashboard/stats` - Get dashboard statistics
- `GET /api/analytics/churn-by-subscription` - Churn by subscription type
- `GET /api/analytics/churn-by-gender` - Churn by gender

### Bulk Processing API
- `POST /api/bulk-predict` - Bulk prediction for multiple customers

## 📊 Model Features

- **Algorithm**: Random Forest Classifier
- **Features**: 8 features including sentiment analysis
- **Scoring**: Risk score, health score, CLV
- **Recommendations**: AI-generated actionable insights

## 🎯 Use Cases

1. **Predictive Analytics**: Identify at-risk customers
2. **Retention Campaigns**: Implement targeted strategies
3. **Customer Insights**: Understand customer behavior
4. **Risk Management**: Prioritize high-risk customers
5. **Performance Tracking**: Monitor retention metrics
6. **Bulk Analysis**: Process large customer datasets

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 📝 Notes

- Make sure model files exist in the `models/` directory
- The app requires trained model files (run `model_training.py` first)
- All predictions are done in real-time
- Charts update automatically with new data

## 🆘 Troubleshooting

### Issue: Module not found
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: Model files not found
**Solution**: Run `python model_training.py` first

### Issue: Port already in use
**Solution**: Change the port in `app.py` (default: 5000)

### Issue: Templates not found
**Solution**: Ensure `templates/` and `static/` directories exist

## 🎉 Features Highlights

✅ **Modern UI** with professional design
✅ **Real-time Predictions** with comprehensive scoring
✅ **Customer Health Scoring** for proactive management
✅ **Predictive CLV** for revenue optimization
✅ **Smart Recommendations** with actionable insights
✅ **Customer Segmentation** for targeted strategies
✅ **Campaign Management** for retention efforts
✅ **Bulk Processing** for large datasets
✅ **Export Capabilities** for data analysis
✅ **Interactive Charts** for visual insights

---

**Built with ❤️ using Flask, Bootstrap, and Chart.js**

**Version**: 2.0.0 (Flask Edition)

**Last Updated**: 2024
