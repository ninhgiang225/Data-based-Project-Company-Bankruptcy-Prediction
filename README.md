# Company Bankruptcy Prediction

This project analyzes and predicts company bankruptcies using financial data from the Taiwan Economic Journal (1999-2009). The dataset includes financial performance metrics, with bankruptcy defined based on Taiwan Stock Exchange regulations.

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
  - *[Specify model here, e.g., Logistic Regression, Random Forest, etc.]*  
- **Performance Metrics**:
  - **Accuracy**: *[Add Value]*
  - **Precision (Bankruptcy)**: *[Add Value]*
  - **Recall (Bankruptcy)**: *[Add Value]*
  - **F1-Score**: *[Add Value]*

*Insert performance visualizations (e.g., ROC Curve, Precision-Recall Curve) here.*

---

### Key Predictors of Bankruptcy
1. **Return on Assets (ROA)**: Lower ROA strongly correlates with bankruptcy.
2. **Operating Profit Rate**: Consistently negative values indicate poor financial health.
3. **Borrowing Dependency**: High reliance on borrowed capital is a major red flag.
4. **Equity-to-Liability Ratio**: Lower ratios signify higher financial risk.
5. **Net Income to Stockholder’s Equity**: Persistent low returns signal inefficiency and risk.

*Insert feature importance graphs or visualizations here.*

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
