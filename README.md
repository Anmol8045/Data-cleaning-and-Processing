       # Data-cleaning-and-Processing

 Step 1: Understand Your Data
Load and Inspect the Data
Use tools like pandas to load and inspect your dataset.

python
Copy
Edit
import pandas as pd

df = pd.read_csv('your_data.csv')
print(df.head())
print(df.info())
Key things to check:
Data types (numeric, categorical, etc.)

Missing values

Inconsistent formatting

Outliers or anomalies

🔹 Step 2: Handle Missing Values

Common strategies:
Remove rows/columns with too many missing values

Impute values using mean/median (numerical) or mode (categorical)

python
Copy
Edit
# Drop missing rows
df.dropna(inplace=True)

# Fill missing numeric with mean
df['column'] = df['column'].fillna(df['column'].mean())


🔹 Step 3: Handle Categorical Data
Convert categorical variables into numbers
Label Encoding (for ordinal categories)

One-Hot Encoding (for nominal categories)

python
Copy
Edit
# One-hot encoding
df = pd.get_dummies(df, columns=['category_column'])

🔹 Step 4: Normalize or Standardize Features
Why?
Many ML algorithms perform better when features are on a similar scale.

python
Copy
Edit
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

🔹 Step 5: Remove Outliers
Use statistical methods or visualizations (e.g., IQR method, Z-score, boxplots).

python
Copy
Edit
from scipy import stats

df = df[(np.abs(stats.zscore(df.select_dtypes(include='number'))) < 3).all(axis=1)]


🔹 Step 6: Feature Engineering
Create new features from existing data that might help your model.

Example: Extracting year from a date column:

python
Copy
Edit
df['year'] = pd.to_datetime(df['date_column']).dt.year


🔹 Step 7: Split the Data
Split the dataset into training and testing (or validation) sets.

python
Copy
Edit
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Tools to Explore:
Pandas & NumPy – Data manipulation

Scikit-learn – Preprocessing and modeling

Matplotlib & Seaborn – Visualization for EDA

Would you like a hands-on example using a real dataset (like Titanic or Iris), or help cleaning your own data?
