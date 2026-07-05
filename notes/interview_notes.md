Baseline Logistic Regression achieved ROC-AUC of 0.772.

For an Early Warning System, recall was prioritised over precision because missing a true runoff customer (false negative) is potentially more costly than issuing an unnecessary warning (false positive).

The baseline model successfully identified approximately 70% of runoff events.

A key finding from the project was that domain-driven feature engineering improved model performance more significantly than model complexity.

By introducing behavioural and relationship-based features such as BalanceSalaryRatio, BalancePerProduct and HighBalanceInactive, ROC-AUC increased from 0.777 to 0.810 without changing the underlying algorithm.

This highlighted the importance of business understanding and behavioural modelling in banking analytics.
