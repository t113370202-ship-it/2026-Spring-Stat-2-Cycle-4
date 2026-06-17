# Statistics Internship Personal Reort
Name: [方立謙]
Student ID: [113370202]

## Summary


## Presentation Video
https://youtu.be/RStr9fVWjfc

YRBS 2007 BMI Analysis Project

This repository contains the dataset preprocessing pipeline, statistical modeling code, and diagnostic figures for the 2007 National Youth Risk Behavior Survey (YRBS) project.

1. Research Question
Do Age and Biological Sex significantly predict Body Mass Index (BMI) among US high schoolers, and does the BMI growth velocity differ between biological boys and girls?

2. Methodology & Variables
Dependent Variable (Y): BMI ($kg/m^2$), derived from Height and Weight.
Independent Variables (X): Age (12 to 18) and Biological Sex (Female = Baseline).
Sampling Weights: Handled via Weighted Least Squares (WLS) as required by CDC guidelines.
Analytical Sample Size: 12,143 records (after removing missing data and BMI outliers < 12 or > 45).

3. Core Equations & Findings
Main Effects Model:
$$\text{BMI} = 13.8241 + (0.4512) \times \text{Age} + (0.3214) \times \text{Is\_Male}$$
Each additional year of age adds 0.45 kg/m² to BMI. Biological males average 0.32 kg/m² higher than females ($p < 0.001$).
Interaction Model:
$$\text{BMI} = 14.1205 + (0.3541) \times \text{Age} - (1.1245) \times \text{Male} + (0.1250) \times (\text{Age} \times \text{Male})$$
Biological boys experience a significantly faster BMI growth velocity (0.48 kg/m² per year) compared to biological girls (0.35 kg/m² per year) during puberty ($p < 0.05$).
