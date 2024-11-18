# Overfitting
This project is based on our previous IMWUT paper [**A Reproducible Stress Prediction Pipeline Using Mobile Sensor Data**](https://dl.acm.org/doi/abs/10.1145/3678578).
https://github.com/SteinPanyu/IndependentReproducibility
https://github.com/SteinPanyu/DeepStressReproducibility
https://github.com/SteinPanyu/IndependentReproducibility_MTL

In the previous study, we observe the existence of overfitting phenomeon and its potential link to personal difference/ distribution shift.

In this work, we try to divide deep into the personal difference within single dataset. 

- What is the within-dataset personal difference?
- What exact personal difference is impacting the generalizability of mobile sensing?
- While the generalized mobile sensing does not work well on unseen users even within-dataset, there is still performance fluctuation across different users. What is the difference between high and low performance users in generalized mobile sensing?

In order to answer these three research questions, we will use three datasets collected from same institution across years and open those datasets in this work. As can be found in the repositories, we divide the codebase based on datasets D-2, D-3, D-4. 

For each dataset, in order to cover the scope of multimodal mobile sensing, we selected ESM stress, calories consumed, and phone usage as our labels. The label specific codes are stored in each respective folder such as ESM_Stress_Prediction_Specific.
