 📊 Project Title: Equity & Learning Gaps Dashboard

👥 Team Members:
Roaa Hajj Chehade
Ghada Al danab
Line Faour

📌 Problem Statement
Education policymakers and teachers often lack clear, visual insights into:
- Where students are underperforming
- Which groups (e.g., boys vs. girls) face inequities
This dashboard provides actionable visual analytics to target interventions and allocate resources fairly.

💡 Proposed Solution
Briefly explain your AI/ML approach.
Mention whether you use supervised, unsupervised, multimodal, or graph-based ML.
State what makes your approach innovative.

Technical Plan
📂 Dataset
We use **synthetic student assessment data**, aggregated into the following CSVs (under `data_proc/`):
- `agg_region_grade_subject.csv` → Subject-level averages per region & grade  
- `agg_region_grade_subject_domain.csv` → Domain-level averages (e.g., Reading, Writing)  
- `agg_gender.csv` → Gender gaps per region & subject  
- `agg_gender_domain.csv` → Gender gaps at domain-level  
- `overlap.csv` → Priority cohorts where learning + equity gaps overlap  
- `overlap_domain.csv` → Domain-level priority cohorts  
Each dataset contains metrics like:
- `avg_score` → average student score  
- `pct_below` → % of students below proficiency (50%)  
- `gap_pp` → gap in percentage points between groups  
- `p_value` → statistical test result  

🏗️ Approach & Architecture
- **Preprocessing:** Raw student rows aggregated → CSVs above  
- **Frontend:** Streamlit multipage app with tabs:
  1. **Overview** → national snapshot  
  2. **Learning Gaps** → heatmap & worst regions  
  3. **Equity** → gender disparities (with domain-level breakdown)  
  4. **Overlap** → priority cohorts where both gaps align  


## 💻 How to Run Locally
```bash
# Clone repo
git clone https://github.com/<your-username>/hackathon-dashboard.git
cd hackathon-dashboard

# Create virtualenv
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
