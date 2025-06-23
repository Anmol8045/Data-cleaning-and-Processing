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

EXAMPLE FOR DATA CLEANING AND PROCESSING

 1. Understand the Data
 2. 
You have columns like:

PassengerId – Unique ID

Survived – Target variable (0 = No, 1 = Yes)

Pclass – Ticket class (1st, 2nd, 3rd)

Name – Passenger name (may contain title info)

Sex – Male/female

Age – Age (has missing values)

SibSp – Siblings/spouses aboard

Parch – Parents/children aboard

Ticket – Ticket number

Fare – Ticket fare (some missing)

Cabin – Cabin number (many missing)

Embarked – Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

✅ 2. Basic Cleaning Steps

A. Drop Irrelevant or Too Sparse Columns
Drop PassengerId, Name, and Ticket (not predictive).

Drop Cabin due to too many missing values (optional).

python
Copy
Edit
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
B. Handle Missing Values
- Age: fill with median or use model-based imputation.
python
Copy
Edit
df['Age'].fillna(df['Age'].median(), inplace=True)
- Embarked: fill with mode (most frequent).
python
Copy
Edit

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

- Fare: if any missing, fill with median.
- 
python
Copy
Edit

df['Fare'].fillna(df['Fare'].median(), inplace=True)
C. Convert Categorical to Numeric
- Sex: encode to 0/1
- 
python
Copy
Edit
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
- Embarked: one-hot encode or label encode
python
Copy
Edit
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
D. Feature Engineering (Optional)
You can extract titles from names if you're keeping the Name column:

python
Copy
Edit
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')

df['Title'] = df['Title'].replace(['Mme'], 'Mrs')

df['Title'] = df['Title'].apply(lambda x: 'Rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)

df = pd.get_dummies(df, columns=['Title'], drop_first=True)

E. Normalize Continuous Values (optional, useful for some models)

Standardize Age, Fare, etc. if using models sensitive to scale:

python
Copy
Edit
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

📦 3. Final Step: Train-Test Split

Split into features and labels:

python
Copy
Edit
from sklearn.model_selection import train_test_split

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

🧠 4. Ready for Machine Learning!

You can now feed X_train and y_train into any classifier, such as:

python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)


