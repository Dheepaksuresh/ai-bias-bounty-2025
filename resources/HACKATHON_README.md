# 📂 AI Bias Bounty Hackathon

Welcome to the **2025 AI Bias Bounty Hackathon**!

Join us as we explore, detect, and report biases in AI models and datasets. This is an exciting opportunity to contribute to ethical AI while building your skills and network.


# 🚀 Overview

The **AI Bias Bounty Hackathon** challenges participants to build machine learning models and generate technical reports that identifies bias within provided datasets. The goal is to encourage the development of fair and responsible AI systems.

**Devpost**: [AI Bias Bounty Hackathon on Devpost](https://ai-bias-bounty-hackathon.devpost.com/)

**Official Website**: [Hack the Fest](https://hackthefest.com/)


# 🗓️ Key Dates

| Schedule                        | Date                               |
| ------------------------------- | ---------------------------------- |
| Registration Period             | *\[June 4 – June 27, 2025]*        |
| Kickoff Event                   | *\[June 28, 2025]*                 |
| Onboarding Period               | *\[June 28 – June 30, 2025]*       |
| Hackathon Launch                | *\[July 1 – July 3, 2025]*   |
| Submission Deadline             | *\[July 3, 2025 11:59pm CST]*      |
| Judging Period                  | *\[July 5 – July 15, 2025]*        |
| Winners Announced               | *\[July 17, 2025]*                 |


# 🎯 Objectives

Participants will:

- Build AI models to analyze and detect bias in provided datasets.

- Generate detailed, well-structured technical reports documenting bias detection.

- Present solutions that contribute to fairness and accountability in AI.


# 🛠️ Getting Started

### 1. Register

- Sign up on our [official website](https://hackthefest.com/) or [Devpost](https://ai-bias-bounty-hackathon.devpost.com/) to officially enter the hackathon.

### 2. Dataset

- You will receive access to the dataset upon registration.

### 3. Deliverables

- `loan_model.py` — Python script containing
    
    - Data cleaning and preprocessing steps
    - Feature engineering (e.g., encoding, binning)
    - Model training (you may use Logistic Regression, Random Forest, XGBoost, or other classification models)
    - Fairness auditing and bias detection
    - Well-commented, readable code

- `submission.csv` — Model's output on the provided test dataset

    - A 2-column CSV:
      - `ID` – test set identifier
      - `LoanApproved` – predicted value (0 or 1)

- Detailed technical report (`ai_risk_report.docx` or `.pdf` or `.md`) using the provided AI Risk Report template

- Visual Evidence of Bias: Submit one or both of the following:

    - `bias_visualization.png` — a clear, readable graphic that illustrates discovered bias
    - Or a `chart/folder` containing:
      - Approval rate bar plots by demographic
      - SHAP/LIME feature importance charts
      - False positive/negative disparities
      - Any visual insights related to model behavior or group fairness
    
    Label every chart clearly. These visuals will help judges understand your biased insights at a glance.

- (Optional) `loan_model.ipynb` — Clean and reproducible Jupyter Notebook

    - Include EDA, model pipeline, audits, and final results
    - Clear markdown explanations and cell comments encouraged

- `README.md` — Describes:

    - The problem you addressed
    - Summary of your model approach and fairness considerations
    - Instructions to run the project & tools/libraries used
    - GitHub repo should be public and well-structured

# 📑 Submission Guidelines

1. Submissions is made on GitHub.

2. Include:

    - Source code

    - Output file

    - Technical report

    - Demo video (required)

3. Follow all provided instructions and deadlines.

4. After completing your GitHub repo, you must submit the repository link using the official [Final Submission Form](https://forms.gle/ES3CY59jEjdaqCvBA). This is how your entry is registered for judging.


# 🏆 Judging Criteria

**1. Accuracy of Bias Identification (30%)**

How effectively does the submission identify multiple types of bias, provide supporting evidence, and handle potential errors?

**2. Model Design and Justification (20%)**

How appropriate is the chosen model for the task, and how well is the design and reasoning behind it explained?

**3. Coverage of Bias Types (15%)**

To what extent does the submission address a wide range of bias categories, from minimal to comprehensive coverage?

**4. Interpretability and Insight (15%)**

How clearly and insightfully does the submission explain the identified biases, especially through tools, visualizations, or interpretability techniques?

**5. Mitigation Suggestions or Solutions (10%)**

Does the submission offer thoughtful and practical suggestions to reduce or mitigate the identified biases, including improvements to the model?

**6. Presentation and Clarity (10%)**

How clearly is the submission presented through visuals, written documentation, and/or demonstrations?

# 📣 Stay Connected

Follow us for updates and highlights:

- **Website**: [Hack the Fest](https://hackthefest.com/)

- **Devpost**: [AI Bias Bounty Hackathon](https://ai-bias-bounty-hackathon.devpost.com/)


# 📚 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all participants. Please review our [Code of Conduct](./CODE_OF_CONDUCT.md).


# 💬 Communication

Join our community via our [Slack Invite Link](https://join.slack.com/t/hackthefest/shared_invite/zt-380la7fd3-xk~zDvk~kZIrqr_HznLHbQ)