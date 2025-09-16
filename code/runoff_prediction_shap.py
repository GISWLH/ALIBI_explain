import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Arial'

def load_and_prepare_data():
    df = pd.read_csv('geo.csv')
    
    feature_cols = [col for col in df.columns if col not in ['time', 'runoff_sum']]
    X = df[feature_cols]
    y = df['runoff_sum']
    
    return X, y, feature_cols

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance - MSE: {mse:.6f}, R2: {r2:.4f}")
    
    return model, scaler, X_train_scaled, X_test_scaled, y_test, y_pred

def explain_with_shap(model, X_train_scaled, X_test_scaled, feature_cols):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled[:20])
    
    shap.summary_plot(shap_values, X_test_scaled[:20], feature_names=feature_cols, show=False)
    plt.savefig('figure/runoff_shap_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    shap.summary_plot(shap_values, X_test_scaled[:20], feature_names=feature_cols, 
                      plot_type="bar", show=False)
    plt.savefig('figure/runoff_shap_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return shap_values

if __name__ == "__main__":
    print("Loading and preparing data...")
    X, y, feature_cols = load_and_prepare_data()
    
    print("Training runoff prediction model...")
    model, scaler, X_train_scaled, X_test_scaled, y_test, y_pred = train_model(X, y)
    
    print("Generating SHAP explanations...")
    shap_values = explain_with_shap(model, X_train_scaled, X_test_scaled, feature_cols)
    
    print("\nScript completed successfully!")