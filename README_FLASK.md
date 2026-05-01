# 🎯 Customer Churn Prediction Platform - Flask Edition

## 🌟 Overview

A comprehensive, AI-powered customer churn prediction and retention platform built with Flask. This platform helps businesses identify at-risk customers, implement proactive retention strategies, and maintain strong customer relationships.

## ✨ Key Features

### 🎨 Modern Professional UI
- **Beautiful gradient theme** with professional teal/blue color scheme
- **Responsive design** that works seamlessly on desktop, tablet, and mobile
- **Smooth animations** with AOS (Animate On Scroll) library
- **Interactive charts** using Chart.js for data visualization
- **Modern Bootstrap 5** components for a polished look

### 🔮 Advanced Analytics

#### 1. **Customer Health Scoring**
   - Real-time health score calculation (0-100 scale)
   - Health categories: Excellent, Good, Fair, Poor
   - Multi-factor health assessment including:
     - Churn probability
     - Customer tenure
     - Engagement level
     - Sentiment analysis
     - Subscription value

#### 2. **Predictive CLV (Customer Lifetime Value)**
   - Calculate expected customer lifetime value
   - Risk-adjusted CLV predictions
   - Revenue optimization insights
   - Expected lifetime in months calculation

#### 3. **Comprehensive Risk Assessment**
   - Multi-factor risk scoring (0-100 scale)
   - Risk categories: Low, Medium, High
   - Visual risk indicators with color coding
   - Factors considered:
     - Churn probability
     - Last interaction days
     - Customer tenure
     - Monthly charges
     - Sentiment analysis

#### 4. **Smart Recommendations**
   - AI-generated actionable recommendations
   - Priority-based action items (Critical, High, Medium, Low)
   - Personalized retention strategies
   - Context-aware suggestions

#### 5. **Customer Segmentation**
   - Automatic customer categorization
   - Risk-based segmentation
   - Segment-specific strategies
   - Bulk customer analysis

#### 6. **Campaign Management**
   - Create retention campaigns
   - Target specific customer segments
   - Campaign templates
   - Email template library

### 📊 Real-time Analytics
- Dashboard with key metrics
- Churn distribution charts
- Subscription type analysis
- Gender-based analytics
- Trend analysis
- Interactive visualizations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository** (if applicable)
   ```bash
   cd customer-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files exist**
   - Run `python model_training.py` if model files don't exist
   - Model files should be in the `models/` directory

4. **Run the application**

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

5. **Access the application**
   - Open your browser
   - Go to: `http://localhost:5000`

## 📁 Project Structure

```
customer-churn-prediction/
├── app.py                      # Flask application (main entry point)
├── templates/                  # HTML templates
│   ├── base.html              # Base template with navigation
│   ├── dashboard.html         # Dashboard page
│   ├── predict.html           # Churn prediction page
│   ├── analytics.html         # Analytics page
│   ├── customers.html         # Customer insights page
│   ├── retention.html         # Retention strategies page
│   └── campaigns.html         # Campaign management page
├── static/                     # Static files
│   ├── css/
│   │   └── style.css          # Custom CSS styles
│   ├── js/
│   │   └── main.js            # JavaScript utilities
│   └── images/                # Image assets
├── models/                     # ML model files
│   ├── churn_model.pkl
│   ├── churn_model_tuned.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_list.pkl
├── data/                       # Data files
│   └── customer_churn.csv
├── requirements.txt            # Python dependencies
├── run_flask.bat              # Windows run script
├── run_flask.sh               # Linux/Mac run script
└── FLASK_README.md            # Detailed documentation
```

## 🎯 Application Pages

### 🏠 Dashboard
- Real-time metrics and statistics
- Key performance indicators
- Interactive charts
- Quick actions
- Platform features overview

### 🔮 Predict Churn
- Single customer prediction
- Risk and health scoring
- CLV calculation
- Sentiment analysis
- Actionable recommendations
- Visual risk gauges
- Probability charts

### 📈 Analytics
- Comprehensive customer analytics
- Churn by subscription type
- Churn by gender
- Trend analysis
- Interactive charts and graphs

### 👥 Customer Insights
- Bulk customer analysis
- Customer segmentation
- Risk categorization
- Health scoring
- Export capabilities
- CSV file upload

### 💡 Retention Strategies
- Retention framework by risk level
- Email templates library
- Action plans
- Best practices
- Template customization

### 📊 Campaign Management
- Create retention campaigns
- Target customer segments
- Campaign templates
- Campaign tracking (coming soon)

## 🔧 API Endpoints

### Prediction API
- `POST /api/predict` - Predict churn for a single customer
  - Request: JSON with customer data
  - Response: Prediction results with scores and recommendations

### Analytics API
- `GET /api/dashboard/stats` - Get dashboard statistics
- `GET /api/analytics/churn-by-subscription` - Churn by subscription type
- `GET /api/analytics/churn-by-gender` - Churn by gender

### Bulk Processing API
- `POST /api/bulk-predict` - Bulk prediction for multiple customers
  - Request: CSV file upload
  - Response: Array of prediction results

## 🎨 Color Scheme

The platform uses a professional color scheme optimized for clarity and visual appeal:

- **Primary**: Blue gradient (#667eea to #764ba2)
- **Success**: Green (#198754)
- **Warning**: Yellow/Orange (#ffc107)
- **Danger**: Red (#dc3545)
- **Info**: Teal/Blue (#0dcaf0)
- **Background**: Light gray (#f8f9fa)

## 📊 Model Features

- **Algorithm**: Random Forest Classifier
- **Features**: 8 features including sentiment analysis
- **Scoring Systems**:
  - Churn Probability
  - Risk Score (0-100)
  - Health Score (0-100)
  - Customer Lifetime Value (CLV)
- **Recommendations**: AI-generated actionable insights

## 🎯 Use Cases

1. **Predictive Analytics**: Identify at-risk customers before they churn
2. **Retention Campaigns**: Implement targeted retention strategies
3. **Customer Insights**: Understand customer behavior and patterns
4. **Risk Management**: Prioritize high-risk customers for intervention
5. **Performance Tracking**: Monitor retention metrics and effectiveness
6. **Bulk Analysis**: Process large customer datasets efficiently
7. **Campaign Management**: Create and manage retention campaigns
8. **Health Monitoring**: Track customer health scores over time

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

**Using Gunicorn:**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Using Docker:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

**Using Heroku:**
```bash
heroku create your-app-name
git push heroku main
```

## 📝 Configuration

### Environment Variables
- `FLASK_ENV`: Development or production
- `SECRET_KEY`: Secret key for sessions (change in production)
- `PORT`: Port number (default: 5000)

### Model Configuration
- Ensure model files are in the `models/` directory
- Model files are loaded automatically on startup
- Fallback to regular model if tuned model is not available

## 🆘 Troubleshooting

### Issue: Module not found
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: Model files not found
**Solution**: Train the model first
```bash
python model_training.py
```

### Issue: Port already in use
**Solution**: Change the port in `app.py`
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: Templates not found
**Solution**: Ensure `templates/` directory exists with all HTML files

### Issue: Static files not loading
**Solution**: Ensure `static/` directory exists with CSS and JS files

## 🎉 Feature Highlights

✅ **Modern UI** with professional design and animations
✅ **Real-time Predictions** with comprehensive scoring
✅ **Customer Health Scoring** for proactive management
✅ **Predictive CLV** for revenue optimization
✅ **Smart Recommendations** with actionable insights
✅ **Customer Segmentation** for targeted strategies
✅ **Campaign Management** for retention efforts
✅ **Bulk Processing** for large datasets
✅ **Export Capabilities** for data analysis
✅ **Interactive Charts** for visual insights
✅ **Responsive Design** for all devices
✅ **Professional Color Scheme** for better UX

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced ML models (XGBoost, LightGBM)
- [ ] Automated email campaigns
- [ ] API endpoints for integration
- [ ] Mobile app support
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] A/B testing framework
- [ ] Integration with CRM systems
- [ ] Customer journey mapping
- [ ] Social media sentiment tracking

## 📧 Support

For issues, questions, or suggestions, please refer to the documentation or create an issue.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

- Your Name/Team

## 🙏 Acknowledgments

- Flask for the amazing web framework
- Bootstrap for the UI components
- Chart.js for beautiful visualizations
- Scikit-learn for ML algorithms
- TextBlob for sentiment analysis

---

**Built with ❤️ using Flask, Bootstrap, and Chart.js**

**Version**: 2.0.0 (Flask Edition)

**Last Updated**: 2024

---

## 🎯 Quick Reference

### Running the Application
```bash
# Windows
run_flask.bat

# Linux/Mac
./run_flask.sh

# Manual
python app.py
```

### Accessing the Application
```
http://localhost:5000
```

### Key URLs
- Dashboard: `/`
- Predict: `/predict`
- Analytics: `/analytics`
- Customers: `/customers`
- Retention: `/retention`
- Campaigns: `/campaigns`

### API Endpoints
- Predict: `POST /api/predict`
- Dashboard Stats: `GET /api/dashboard/stats`
- Bulk Predict: `POST /api/bulk-predict`

---

**Enjoy your enhanced Customer Churn Prediction Platform! 🚀**
