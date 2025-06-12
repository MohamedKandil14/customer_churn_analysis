import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

print("تم استيراد المكتبات بنجاح!")

try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("\nتم تحميل البيانات بنجاح! إليك أول 5 صفوف:")
    print(df.head())
except FileNotFoundError:
    print("خطأ: لم يتم العثور على ملف 'WA_Fn-UseC_-Telco-Customer-Churn.csv'.")
    print("تأكد من تنزيله من Kaggle ووضعه في نفس المجلد الذي يوجد به ملف 'churn_analysis.py'.")
    exit()

print(f"\nعدد الصفوف والأعمدة الأصلية: {df.shape}")

print("\n------------------------------------")
print("معلومات أولية عن البيانات (df.info()):")
df.info()

print("\n------------------------------------")
print("ملخص إحصائي للبيانات الرقمية (df.describe()):")
print(df.describe())

print("\n------------------------------------")
print("عدد القيم المفقودة في كل عمود (df.isnull().sum()):")
print(df.isnull().sum())

print("\n------------------------------------")
print("بدء تنظيف البيانات ومعالجتها...")

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
print("\nتم معالجة عمود 'TotalCharges' والقيم المفقودة.")
print(f"عدد الصفوف والأعمدة بعد حذف القيم المفقودة: {df.shape}")

df.drop('customerID', axis=1, inplace=True)
print("تم حذف عمود 'customerID'.")

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("تم تحويل عمود 'Churn' إلى 0 و 1.")

df = pd.get_dummies(df, columns=[
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
], drop_first=True, dtype=int)

print("\nتم تطبيق One-Hot Encoding على الأعمدة الفئوية.")
print("\nأول 5 صفوف بعد التنظيف والمعالجة:")
print(df.head())
print(f"\nشكل البيانات النهائي بعد التنظيف والمعالجة: {df.shape}")

print("\nتم الانتهاء من تنظيف ومعالجة البيانات بنجاح.")

print("\n------------------------------------")
print("بدء التحليل الاستكشافي للبيانات (EDA)...")

plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='coolwarm')
plt.title('Distribution of Customer Churn (0=No Churn, 1=Churn)')
plt.xlabel('Churn Status')
plt.ylabel('Number of Customers')
plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn'])
plt.show()

churn_rate = df['Churn'].value_counts(normalize=True) * 100
print(f"\nنسبة العملاء الذين غادروا (Churn Rate):\n{churn_rate}")

print("\nتحليل علاقة Churn ببعض المتغيرات الهامة:")

plt.figure(figsize=(8, 5))
sns.countplot(x='InternetService_Fiber optic', hue='Churn', data=df, palette='viridis')
plt.title('Churn by Internet Service Type (Fiber Optic vs. Others)')
plt.xlabel('Internet Service (0=DSL/No Internet, 1=Fiber Optic)')
plt.ylabel('Number of Customers')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Contract_Two year', hue='Churn', data=df, palette='magma')
plt.title('Churn by Contract Type (Two Year vs. Others)')
plt.xlabel('Contract (0=Month-to-month/One year, 1=Two year)')
plt.ylabel('Number of Customers')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, palette='coolwarm')
plt.title('Churn Distribution by Monthly Charges')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Number of Customers')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, palette='coolwarm')
plt.title('Churn Distribution by Tenure')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.show()

plt.figure(figsize=(15, 12))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of All Features')
plt.show()

print("\nأهم الارتباطات مع Churn (الارتباطات القوية بالقرب من 1 أو -1):")
print(corr_matrix['Churn'].sort_values(ascending=False))

print("\nتم الانتهاء من التحليل الاستكشافي للبيانات.")

print("\n------------------------------------")
print("بدء نمذجة التعلم الآلي...")

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nشكل بيانات التدريب (X_train): {X_train.shape}, شكل الهدف للتدريب (y_train): {y_train.shape}")
print(f"شكل بيانات الاختبار (X_test): {X_test.shape}, شكل الهدف للاختبار (y_test): {y_test.shape}")

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = StandardScaler()

X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

print("\nتم تطبيق Standard Scaling على المتغيرات الرقمية.")

model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train)

print("\nتم تدريب نموذج Logistic Regression بنجاح.")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n------------------------------------")
print("تقييم أداء النموذج:")
print(f"Accuracy (الدقة الإجمالية): {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision (الدقة لـ Churn=1): {precision_score(y_test, y_pred):.2f}")
print(f"Recall (الاستدعاء لـ Churn=1): {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score (متوسط الدقة والاستدعاء): {f1_score(y_test, y_pred):.2f}")

print("\nConfusion Matrix (مصفوفة الارتباك):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report (تقرير التصنيف - مفصل):")
print(classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("\nتم الانتهاء من نمذجة التعلم الآلي وتقييم النموذج.")
print("\n------------------------------------")
print("التحليل اكتمل بنجاح! راجع المخرجات والرسوم البيانية.")