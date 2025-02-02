This repository contains all the work produced from the collaboration between Criteo and Columbia University for the Data Science Capstone &amp; Ethics (ENGI E4800), Spring 2024 semester

Columbia University contributors: Xinyi Chen, Martin Fluder, Jean Law, Ling Lin, Yue Yin

Criteo: [Jan Benzing](https://github.com/janbenzing), [Matt Merenich](https://github.com/mmerenich21)  

## Context:
Criteo (NASDAQ: CRTO) is a worldwide leader in data-driven advertising. Every day, we serve around 4 billion ads, and we have around 20+ millions of clicks. We have various algorithms that help us serve the most appropriate ads that provide a good user experience (UX) to the final user. However, we noticed that some clicks have odd patterns, so-called misclicks. Practice showed that those misclicks are due to slow phones, heavy banners, and misleading websites that spoil UX. Heatmaps can usually spot this. In the case of slow banners, heat maps look completely broken.

Criteo is working on spotting such misclicks and slow banner issues. For that, we’ve developed a p-misclick metric (the probability of a click being a misclick) and a heatmap generation tool (the user can observe the click distribution on a given scope). We want to automatize this approach. Your objective is to build a clusterization/anomaly detection algorithm (i.e., fully unsupervised learning) to predict whether the given heatmap is broken. Thus, we could spot the “hot points” with issues and address them.

## Repository
This repository contains the following directories:
- `datasets`: all datasets provided by Criteo, as well as a train-test split for the first dataset provided
- `reports`: midterm, ethics and final report
- `notebooks`: code used to implement the methods that have been presented in the [final report](https://github.com/fluderm/adtech_broken_banner_detection/blob/main/reports/Capstone_final_report.pdf). Notably, these are the final models, a lot of experimenting is detailed (and unstructured) in 'old code'.

## More Project Details

More details about our approaches and conclusions can be found in our [final](https://github.com/fluderm/adtech_broken_banner_detection/blob/main/reports/Capstone_final_report.pdf) and [midterm](https://github.com/fluderm/adtech_broken_banner_detection/blob/main/reports/Capstone_midterm_report.pdf) reports (including a discussion on [AGI](https://github.com/fluderm/adtech_broken_banner_detection/blob/main/reports/Capstone_AGI_report.pdf) for the ethics portion of the course).

