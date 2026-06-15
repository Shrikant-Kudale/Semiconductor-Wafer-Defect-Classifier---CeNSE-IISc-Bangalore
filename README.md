# ⚡ WaferFab Intelligence Dashboard 🛡️

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-orange?logo=streamlit)](https://semiconductor-wafer-defect-classifier-cense-iisc-bangalore.streamlit.app/)
[![IISc Bengaluru](https://img.shields.io/badge/IISc-CeNSE-blue)](https://cense.iisc.ac.in/)

---
**Live App:** [https://semiconductor-wafer-defect-classifier-cense-iisc-bangalore.streamlit.app/](https://semiconductor-wafer-defect-classifier-cense-iisc-bangalore.streamlit.app/)

## 🚀 Project Overview: The Future of Fab Intelligence
In the high-stakes world of semiconductor manufacturing, the difference between a high-yield production run and a costly failure lies in the ability to detect and diagnose process excursions in real-time. The **WaferFab Intelligence Dashboard** is an industrial-grade machine learning application developed to bridge the gap between complex metrology inspection data and actionable engineering decisions.

This project represents the practical application of data-driven methodologies—specifically **Machine Learning-Based Defect Classification**—to categorize wafer defects automatically, reducing reliance on manual inspection and accelerating root-cause analysis in cleanroom environments.



---

## 🎯 What is this project?
At its core, this project is a **Predictive Diagnostic System**. It takes raw tabular metrology data—measurements collected during the fabrication process such as oxide thickness, critical dimension (CD) variations, and etch uniformity—and feeds them into a highly optimized **XGBoost ensemble model**.

* **The Objective:** To perform multi-class classification of semiconductor defect patterns (e.g., Center-defects, Edge-Rings, Scratches, or Normal wafers).
* **The Deliverable:** A robust, interactive dashboard that not only predicts the *type* of defect occurring on the wafer but also assesses the *confidence level* of that prediction to assist process engineers in making rapid, data-backed interventions.

---

## 🛠️ How does it work?
The dashboard is powered by a sophisticated data pipeline developed to simulate real-world wafer fabrication characteristics.

1.  **Data Simulation & Feature Engineering:** We simulate complex spatial wafer-map patterns alongside paired tabular metrology data. By synthesizing correlations between process parameters (e.g., how a deviation in gas flow during etching manifests as an 'Edge-Ring' defect), the model learns the underlying physics of failure.
2.  **Machine Learning Inference:** The application utilizes a pre-trained **XGBoost Classifier**, chosen for its superior performance on structured/tabular data. The model is wrapped in a `scikit-learn` pipeline with feature scaling to ensure that inputs from varying physical domains (e.g., nanometer measurements vs. percentage-based uniformity) are normalized for maximum accuracy.
3.  **Dynamic UI/UX Architecture:** Using **Streamlit**, the application generates a real-time reactive UI. It captures user inputs via ergonomic sliders, processes them through the cached model, and renders interactive visualizations using **Plotly**. This includes diagnostic Radar Charts for parameter profiling and Probability Distribution bars for classification confidence.

---

## 💡 Why does this matter?
The semiconductor industry is currently facing an unprecedented demand for high-yield, high-reliability chips. This project addresses the "Why" behind the "How":

* **Reducing Yield Loss:** By providing automated classification, fabrication facilities can detect systematic drifts in process tools (like CMP or Lithography scanners) before they result in thousands of dollars of wasted material.
* **Human-in-the-Loop Efficiency:** Instead of asking an engineer to manually inspect thousands of wafer maps, this dashboard filters the "noise" and alerts them to the specific process excursions that require immediate calibration or maintenance.
* **Academic & Industrial Synergy:** This project serves as a capstone to the **Foundation Training Program at CeNSE, IISc**, demonstrating that advanced AI is not just a software tool, but a critical component of modern semiconductor characterization and fabrication.

---

## 🎓 Training Context: CeNSE, IISc Summer School
This application was conceived and refined during the **Ministry of Tribal Affairs (MoTA) Foundation Training Program on Semiconductor Fabrication & Characterization (June 08-19, 2026)**. Our engagement covered the full breadth of semiconductor science:

* **Fabrication Mastery:** Deep dives into Lithography, Thin-Film Deposition, Doping, and Reactive Ion Etching.
* **Computational MEMS:** Hands-on experience with **COMSOL Multiphysics** for resonator and CVD reactor modeling.
* **Characterization:** Understanding the atomic-scale world through SEM/TEM microscopy and advanced electrical characterization.
* **Packaging & Systems:** Analyzing high-speed signal propagation, Altium PCB design, and industrial thermal management.

---

## ⚙️ Dashboard Tech Stack
* **Frontend:** Streamlit 🌐
* **Machine Learning:** XGBoost, Scikit-learn, Joblib 🤖
* **Data Manipulation:** NumPy, Pandas 📊
* **Visualization:** Plotly (Radar Charts, Bar Graphs) 📈
* **Deployment:** Streamlit Community Cloud (CI/CD Integrated) ☁️

---

*Built with 💜 for the MoTA Foundation Training Program at CeNSE, IISc Bengaluru.*
