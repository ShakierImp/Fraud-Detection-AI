# FraudGuardian AI – Streamlit UX Best Practices

This guide provides UI/UX optimization tips for the FraudGuardian AI demo dashboard.

---

## 1. First Impressions
- **📂 Clear Call-to-Action:** Place `st.file_uploader()` prominently at the top center with text: *“Upload your transactions CSV to begin fraud detection.”*  
- **👀 Visual Hierarchy:** Use `st.title()` for project name → `st.subheader()` for short description → `st.file_uploader()` as the main action.  
- **⏳ Loading States:** Use `st.spinner("Analyzing transactions...")` and `st.progress()` bars during model inference to reassure users.  

---

## 2. Visual Design for Fraud Detection
- **🎨 Color Scheme:**  
  - High risk = **Red (#E74C3C)**  
  - Medium risk = **Amber (#F39C12)**  
  - Low risk = **Green (#27AE60)**  
- **⚠️ Icons:**  
  - High = `:warning:`  
  - Medium = `:information_source:`  
  - Low = `:white_check_mark:`  
- **✍️ Typography & Spacing:** Use Streamlit’s default font but emphasize hierarchy with `st.markdown("### Section Title")` and adequate whitespace (`st.write("")` for spacing).  

---

## 3. Highlighting Flagged Transactions
- **📑 Table Styling:** Use `st.dataframe()` with conditional formatting (`df.style.applymap`) to color-code rows by risk level.  
- **🔍 Sorting & Filtering:** Add sidebar filters (`st.sidebar.selectbox()`) for risk category, amount thresholds, or time period.  
- **💡 Hover Effects:** Provide tooltips using `st.tooltip()` (or embed notes in `st.markdown()`) to explain fraud score meaning.  

---

## 4. Data Visualization
- **📊 Chart Types:**  
  - Risk distribution → `st.bar_chart()` or `st.altair_chart()` histogram.  
  - Transaction timelines → `st.line_chart()`.  
  - Geographic anomalies → `st.map()`.  
- **📈 Confidence Scores:** Represent with **horizontal bar charts** (Altair) where bar length = fraud probability.  
- **🖱️ Interactivity:** Allow filtering charts dynamically (`st.multiselect()` for merchants, time ranges).  

---

## 5. Mobile Considerations
- **📱 Responsive Layout:** Use `st.columns([2,1])` instead of wide multi-column grids for clarity on laptops.  
- **🔄 Adaptive Text:** Keep text concise—avoid long paragraphs.  
- **📌 Fixed Actions:** Keep “Upload File” and “Run Analysis” buttons always visible (sticky positioning not native, but repeat at bottom of page).  

---

## 6. Demo Flow (Hackathon Journey)
1. **👋 Intro Screen:** Title + tagline + file uploader.  
2. **📂 Upload Transactions:** Judges upload sample file (or use preloaded demo button).  
3. **⏳ Show Progress:** Display spinner → “Analyzing 10,000 transactions…”  
4. **🚨 Results Summary:** Big numbers: *“Detected 132 suspicious transactions (1.3%).”*  
5. **📊 Visual Highlights:**  
   - Fraud risk distribution chart.  
   - Confidence score bar chart.  
   - Top 10 flagged transactions table (with red highlights).  
6. **🔍 Drill Down:** Click/hover to see transaction details (merchant, amount, location).  
7. **✅ Wrap-Up:** End with a “FraudGuardian AI saves time by detecting anomalies instantly.”  

---

✨ **Pro Tip:** Always **guide the judges’ eyes** with consistent colors and step-by-step cues—avoid clutter so they remember the story, not just the dashboard.
