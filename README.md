# Company Bankruptcy Prediction

This project analyzes and predicts company bankruptcies using financial data from the Taiwan Economic Journal (1999-2009). The dataset includes financial performance metrics, with bankruptcy defined based on Taiwan Stock Exchange regulations.

## How to run the code
- Look through the Jupyter notebook ,CompanyBankruptcyPrediction.ipynb, to gain the findings, analysis and key insights. At the end of the notebook, it preview a dashboard
- To run it, run "streamlit run application.py" in the terminal. It will return a streamlit app in your local web
  <img width="683" alt="image" src="https://github.com/user-attachments/assets/ab406261-9ba6-4a24-9e6a-908c09b5fb57" />
- insert financial metrics into that dashboard so that it can predict if the company faces the likelyhood of bankruptcy. It also implies some important financial indicators so that company can modify their plan and business strategy to avoid being bankrupted
<img width="978" alt="image" src="https://github.com/user-attachments/assets/2705e64f-3758-45b8-9121-d1e279d0ab7d" />


## Key Insights and Findings

### Data Overview
- **Dataset Size**: 6,819 companies (1999-2009)
- **Class Distribution**:
  - Non-bankrupt: 6,599 (96.8%)
  - Bankrupt: 220 (3.2%)
- **Key Features**: 
  - Financial ratios such as ROA, operating profit rate, borrowing dependency, and equity-to-liability ratio.
- **Class Imbalance**: 
  - Addressed using SMOTE (Synthetic Minority Oversampling Technique).

---

### Predictive Modeling
- **Preprocessing**:
  - Dataset was clean (no missing values or duplicates).
  - Scaling was unnecessary due to normalized values.
- **Model Used**: 
  - *[Machine learning, Linear regression, Neural network, XGboost, Random Forest, LightGBM, SVM, etc.]*  
- **Performance Metrics**:
  - **Accuracy**: *[94%]*
  - **Precision (Bankruptcy)**: *[98%]*
  - **Recall (Bankruptcy)**: *[100%]*
  - **F1-Score**: *[99%]*

<img width="392" alt="image" src="https://github.com/user-attachments/assets/088996e3-662a-4371-9ec5-0f594f978b96" />

---

### Key Predictors of Bankruptcy
1. **Return on Assets (ROA)**: Lower ROA strongly correlates with bankruptcy.
2. **Operating Profit Rate**: Consistently negative values indicate poor financial health.
3. **Borrowing Dependency**: High reliance on borrowed capital is a major red flag.
4. **Equity-to-Liability Ratio**: Lower ratios signify higher financial risk.
5. **Net Income to Stockholder’s Equity**: Persistent low returns signal inefficiency and risk.

<img width="436" alt="image" src="https://github.com/user-attachments/assets/cfc441bd-0687-4666-bbe3-73d1edf76e2b" />

---

## Limitations
1. **Class Imbalance**:
   - Despite SMOTE, residual imbalance affects precision for predicting bankruptcies.
2. **Feature Selection**:
   - Potential redundancy or collinearity among features.
3. **Time-Specific Data**:
   - Findings may not generalize to current economic conditions.
4. **Data Quality Assumptions**:
   - Relied on the Taiwan Economic Journal’s data accuracy.

---

## Potential Improvements
1. **Advanced Balancing**:
   - Combine SMOTE with under-sampling or ensemble methods.
2. **Feature Engineering**:
   - Explore derived features, such as lagged indicators, for deeper insights.
3. **Model Enhancements**:
   - Investigate deep learning models for complex interactions.
4. **Economic Adjustments**:
   - Update financial ratios for inflation or economic changes.

---

## Conclusion
This project highlights the potential of predictive models to identify at-risk companies using financial data. By leveraging key metrics and addressing class imbalance, the model provides valuable insights into bankruptcy risks. Future work should focus on refining features and extending applicability to broader datasets and time periods.
