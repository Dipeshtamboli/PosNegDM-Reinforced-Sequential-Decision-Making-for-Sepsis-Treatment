# Reinforced Sequential Decision-Making for Sepsis Treatment: The PosNegDM Framework with Mortality Classifier and Transformer

This folder contains the code for the paper "Reinforced Sequential Decision-Making for Sepsis Treatment: The PosNegDM Framework with Mortality Classifier and Transformer."

The paper titled "Sepsis Treatment Using Machine Learning: A Transformer-Based Decision-Making Framework" introduces a novel approach to sepsis treatment by leveraging machine learning techniques. The primary focus of the paper is to address the challenges associated with sepsis management, a life-threatening condition with high mortality rates, by proposing the POSNEGDM framework.

## Using this code

To use this code, you will need to install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

We have provided a sample dataset in the `data` folder for demonstration purposes. You can replace this dataset with the actual data from [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/). We cannot provide the actual data due to licensing restrictions.


Once you have installed the dependencies and prepared the data, you can run the code by executing the following command:

```bash
python train.py
```



## Key contributions and actions of the paper include:
1. **Introduction of POSNEGDM Framework**:
   - Introduces the POSNEGDM framework, which stands for "Reinforcement Learning with Positive and Negative Demonstrations for Sequential Decision-Making."
   - Utilizes a transformer-based model and a mortality classifier to optimize treatment decisions for sepsis patients.
   
2. **Improving Patient Outcomes**:
   - Demonstrates that the POSNEGDM framework significantly enhances patient survival rates, saving a high percentage of patients compared to established machine learning algorithms.
   - Shows promising results in improving sepsis treatment outcomes and potentially reducing healthcare costs.

3. **Personalized Treatment Decisions**:
   - Incorporates individual patient characteristics and expert actions through reinforcement learning to tailor treatment plans.
   - Utilizes a mortality classifier with high accuracy to guide treatment decisions towards positive outcomes.

4. **Comparative Analysis**:
   - Compares the performance of the POSNEGDM framework with existing machine learning algorithms, such as Decision Transformer and Behavioral Cloning, showcasing superior results in terms of patient survival rates.

5. **Future Research Directions**:
   - Discusses the implications of the results and outlines potential avenues for future research in the domain of sepsis treatment.
   - Emphasizes the importance of deploying the model in clinical practice and integrating it with existing healthcare systems for seamless implementation.

Overall, the paper aims to revolutionize sepsis treatment by introducing a data-driven and personalized approach that leverages machine learning to optimize treatment decisions, improve patient outcomes, and contribute to enhanced patient care in the context of sepsis management.

## Citation

```
@article{tamboli2024reinforced,
  title={Reinforced Sequential Decision-Making for Sepsis Treatment: The POSNEGDM Framework with Mortality Classifier and Transformer},
  author={Tamboli, Dipesh and Chen, Jiayu and Jotheeswaran, Kiran Pranesh and Yu, Denny and Aggarwal, Vaneet},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```
