import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fpdf import FPDF

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 1: Data Cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Step 2: Rename columns for clarity
df.rename(columns={
    'Annual Income (k$)': 'Annual_Income',
    'Spending Score (1-100)': 'Spending_Score'
}, inplace=True)

# Step 3: Encode Gender
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Step 4: Select features and scale
features = df[['Gender', 'Age', 'Annual_Income', 'Spending_Score']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 5: Determine optimal clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Optional: Plot Elbow graph
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.savefig("elbow_plot.png")
plt.close()

# Step 6: Apply KMeans with optimal clusters (e.g., 5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 7: Reduce dimensions for visualization using PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
df['PCA1'] = principal_components[:, 0]
df['PCA2'] = principal_components[:, 1]

# Step 8: Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title("Customer Segments based on Purchase Behavior (PCA)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title="Cluster")
plt.savefig("customer_segmentation_plot.png")
plt.close()

# Step 9: Generate PDF Report
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Customer Segmentation Report", ln=True, align="C")

    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)

    def section_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 10, body)
        self.ln()

# Create PDF
pdf = PDFReport()
pdf.add_page()
pdf.section_title("1. Dataset Overview")
pdf.section_body(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns after cleaning.")

pdf.section_title("2. Feature Engineering")
pdf.section_body("Encoded Gender as binary (Male=1, Female=0). Used features: Gender, Age, "
                 "Annual Income, and Spending Score. Data was normalized using StandardScaler.")

pdf.section_title("3. Clustering and PCA")
pdf.section_body("Applied KMeans clustering (k=5) after evaluating the Elbow Method. Used PCA for reducing "
                 "dimensions for 2D visualization.")

pdf.image("customer_segmentation_plot.png", x=10, w=180)

pdf.output("Customer_Segmentation_Report.pdf")

