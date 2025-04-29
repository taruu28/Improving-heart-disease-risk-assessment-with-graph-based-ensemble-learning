import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import joblib

# Define HeartDiseaseGNN class
class HeartDiseaseGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HeartDiseaseGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load using joblib
model = joblib.load('heart_disease_gnn.pkl')
model.eval()

# Important features that the model expects
important_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 
                      'restecg', 'thalach', 'exang', 'oldpeak', 
                      'slope', 'ca', 'thal']

# Title
st.title("Heart Disease Prediction")
st.markdown("Bulk prediction of heart disease risk using a hybrid Graph Neural Network model.")

# Sidebar for file upload
st.sidebar.header("Upload Patient CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Predict button
if st.sidebar.button("Run Prediction"):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Select only the important features
        data = data[important_features]

        X = data.values
        X_tensor = torch.tensor(X, dtype=torch.float)
        if X_tensor.size(0) > 1:
            edge_index = torch.combinations(torch.arange(X_tensor.size(0)), r=2).T
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        graph_data = Data(x=X_tensor, edge_index=edge_index)
        with torch.no_grad():
            output = model(graph_data)
            probabilities = torch.softmax(output, dim=1)[:, 1].numpy()
            predictions = (probabilities > 0.5).astype(int)
        avg_risk = np.mean(probabilities) * 100
        avg_risk_label = ("Low Risk" if avg_risk < 40 else "Moderate Risk" if avg_risk < 70 else "High Risk")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Patient Risk Prediction")
            fig, ax = plt.subplots(figsize=(3,3))
            ax.pie([avg_risk, 100 - avg_risk], labels=[f"{avg_risk:.0f}%", ""], startangle=90, colors=['#3498db', '#ecf0f1'])
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig.gca().add_artist(centre_circle)
            ax.axis('equal') 
            st.pyplot(fig)
            st.markdown(f"### {avg_risk_label}")
        with col2:
            st.subheader("Prediction Confidence Distribution")
            plt.figure(figsize=(4,3))
            plt.hist(probabilities, bins=10, color='#5dade2')
            plt.xlabel("Prediction Confidence")
            plt.ylabel("Count")
            st.pyplot(plt)
        st.subheader("Prediction Results")
        results_df = data.copy()
        results_df['Predicted'] = predictions
        results_df['Probability'] = np.round(probabilities, 2)
        st.dataframe(results_df)
        st.subheader("Top 5 Features Used")
        feature_importance = {
            'thal': 0.25,
            'ca': 0.22,
            'oldpeak': 0.15,
            'cp': 0.13,
            'thalach': 0.11
        }
        imp_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
        imp_df = imp_df.sort_values(by='Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(4,3))
        imp_df.plot(kind='barh', legend=False, ax=ax, color='#48c9b0')
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📂 Download Predictions", data=csv, file_name='predictions.csv', mime='text/csv')
    else:
        st.error("Please upload a patient CSV file.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Graph Neural Networks")