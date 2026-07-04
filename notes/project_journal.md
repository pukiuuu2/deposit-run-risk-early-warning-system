# Deposit Run Risk Early Warning System - Project Journal

## Session 1 - Business Design

### Why this project?

Most banking portfolio projects focus on customer churn prediction. However, from a liquidity risk perspective, customer attrition can be interpreted as a proxy for deposit runoff events.

Therefore, this project aims to develop an Early Warning System (EWS) to identify customers at risk of deposit withdrawal and support proactive liquidity risk management.

### Initial Business Hypotheses

1. Customers with higher balances are more likely to create material liquidity risk when they leave.
2. Inactive customers have higher runoff risk.
3. Customers using multiple banking products are more sticky and less likely to leave.
4. Relationship tenure is negatively associated with runoff risk.
5. Historical behavioural variables would be stronger predictors than demographic variables if available.

### Potential Feature Ideas

* Balance / Salary
* Balance / Number of Products
* Age Band
* Balance Band
* Customer Value Score

### Dataset Limitations

The Kaggle dataset does not contain:

* Transaction history
* Deposit inflow/outflow trend
* Salary payment history
* Digital banking activity
* Historical account activity

These variables would likely improve model performance in a production banking environment.
