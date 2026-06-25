# ✈️ Passenger Satisfaction Predictive Analytics & Decision Optimization

This repository delivers an end-to-end predictive framework that extracts, evaluates, and optimizes passenger satisfaction parameters using an algorithmic **Decision Tree Classifier** compared against a standard **Logistic Regression baseline**.

## 📊 Business Context & Objective
Understanding passengers' experiential triggers is critical to optimizing operational investments. By analyzing raw metrics from **Invistico Airline**, this predictive engine:
- Automatically detects complex, multi-level interactions among cabin amenities.
- Isolates the absolute top operational metrics driving customer loyalty.
- Provides non-technical stakeholders with a clear, auditable tree structure to guide strategic decision-making.

---

## ⚙️ Hyperparameter Tuning Matrix (`GridSearchCV`)
Unconstrained decision trees are prone to overfitting by mapping training data noise. To ensure robust generalization, we applied a cross-validated grid search over three structural hyperparameters:

* **`max_depth`**: Evaluated `[5, 7, 10]` | **Optimal Selected: 10** *Prevents the model from creating overly granular rules while allowing enough complexity to capture interaction patterns.*
* **`min_samples_split`**: Evaluated `[10, 20, 30]` | **Optimal Selected: 20** *Ensures an internal node will only split if it contains at least 20 sample observations.*
* **`min_samples_leaf`**: Evaluated `[5, 10, 15]` | **Optimal Selected: 5** *Guarantees terminal leaf nodes maintain at least 5 samples, smoothing predictions at the outer edges of the tree.*

---

## 🏆 Performance Profile Breakdown

Below is the comparative test evaluation matrix separating the non-linear Decision Tree from the parametric Logistic Regression baseline:

| Key Metric Indicator | Optimized Decision Tree | Baseline Logistic Regression | Operational Advantage |
| :--- | :---: | :---: | :--- |
| **F1-Score (Satisfied Class)** | **92.70%** | **84.36%** | **+8.34% Net Accuracy Improvement** |
| **Precision (Satisfied)** | 91.54% | 83.12% | Fewer false positives; reduces misallocated promotional spend. |
| **Recall (Satisfied)** | 93.89% | 85.64% | Better identification of truly satisfied brand advocates. |
| **Model Transparency** | **High** (Visual Pathways) | **Medium** (Log-Odds Weights) | Clear, actionable "If-Then" operational rules. |

---

## 🚀 Top Operational Service Driver Rankings

The optimization process quantifies feature importance through a **Gini Importance Coefficient (Information Gain)**. Capital deployment budgets should be prioritized according to these top-ranked drivers:

1. **In-flight Entertainment (49.17%)** – The single strongest indicator of a positive passenger experience.
2. **Seat Comfort (20.83%)** – Represents a foundational baseline requirement for long-haul traveler satisfaction.
3. **Ease of Online Booking (6.52%)** – Highlights the high impact of friction-free digital touchpoints during ticketing.
4. **Customer Type / Disloyal Accounts (3.25%)** – Flags high churn risks within non-contractual and infrequent flyer cohorts.
5. **Departure/Arrival Time Convenience (2.68%)** – Confirms the ongoing baseline importance of schedule reliability.

---

## 📂 Repository Contents & Implementation Layout

1. **`decision_tree_airline.ipynb`**:
   * Complete data cleansing and median-based imputation pipelines.
   * One-Hot structural encoding parameters.
   * Fully executed `GridSearchCV` hyperparameter tuning loops.
   * Inline execution output displaying the **Confusion Matrix** and **`plot_tree` layout**.
2. **`README.md`**: Comprehensive summary of dataset tuning strategies, key performance metrics, and tactical insights for leadership.

---

## 🛠️ Execution & Quickstart Guide

To replicate this analytic pipeline locally, clone this repository and install the standard data science dependencies:

```bash
# Install core components
pip install pandas numpy matplotlib seaborn scikit-learn

# Launch the notebook workspace
jupyter notebook decision_tree_airline.ipynb
