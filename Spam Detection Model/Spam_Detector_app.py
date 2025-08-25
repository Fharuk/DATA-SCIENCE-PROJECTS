import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# ==============================
# Streamlit Config
# ==============================
st.set_page_config(page_title="Spam SMS Classifier", page_icon="📩", layout="wide")



# ==============================
# Load pipeline
# ==============================
with open("spam_classifier_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)



# ==============================
# Sidebar
# ==============================
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This spam classifier is built with:

- **TF-IDF Vectorizer** for text preprocessing  
- **Random Forest** as the classifier  
- **Accuracy:** ~99% on test data  
""")

# ==============================
# App Title
# ==============================
st.title("📩 Spam SMS Detector")
st.markdown("This app classifies SMS messages as **Spam** or **Ham (Not Spam)** using a trained Machine Learning model.")

# ==============================
# Single Message Prediction
# ==============================
st.subheader("✍️ Enter your SMS message below:")
user_input = st.text_area("Message")

if st.button("🔍 Classify"):
    if user_input.strip() != "":
        prediction = pipeline.predict([user_input])[0]
        prob = pipeline.predict_proba([user_input])[0]

        if prediction == "spam":
            st.error(f"⚠️ This message is classified as **SPAM** with probability {prob[1]:.2f}")
        else:
            st.success(f"✅ This message is classified as **HAM (Not Spam)** with probability {prob[0]:.2f}")
    else:
        st.warning("⚠️ Please enter a message for classification.")

# ==============================
# Bulk Classification via CSV
# ==============================
st.subheader("📂 Upload a CSV file for bulk classification")
uploaded_file = st.file_uploader("Upload CSV file with a column named 'message'", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "message" in data.columns:
        data["prediction"] = pipeline.predict(data["message"])
        data["spam_probability"] = pipeline.predict_proba(data["message"]).tolist()

        st.write("✅ Predictions completed! Here are the first few results:")
        st.dataframe(data.head())

        # Download results
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", data=csv, file_name="classified_messages.csv", mime="text/csv")

        # ==============================
        # Visualizations
        # ==============================
        st.subheader("📊 Insights from Uploaded Data")

        col1, col2 = st.columns(2)

        # 1. Spam vs Ham distribution
        with col1:
            st.markdown("### Spam vs Ham Distribution")
            dist = data["prediction"].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=dist.index, y=dist.values, palette="coolwarm", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel("Message Type")
            st.pyplot(fig)

        # 2. Prediction probability histogram
        with col2:
            st.markdown("### Prediction Probability Distribution")
            probs = [p[1] for p in data["spam_probability"]]
            fig, ax = plt.subplots()
            sns.histplot(probs, bins=20, kde=True, color="red", ax=ax)
            ax.set_xlabel("Spam Probability")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        # 3. WordClouds
        st.markdown("### ☁️ WordClouds of Messages")
        spam_text = " ".join(data[data["prediction"] == "spam"]["message"].astype(str))
        ham_text = " ".join(data[data["prediction"] == "ham"]["message"].astype(str))

        wc_spam = WordCloud(width=600, height=400, background_color="black", colormap="Reds").generate(spam_text)
        wc_ham = WordCloud(width=600, height=400, background_color="black", colormap="Blues").generate(ham_text)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Spam Messages WordCloud**")
            fig, ax = plt.subplots()
            ax.imshow(wc_spam, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        with col4:
            st.markdown("**Ham Messages WordCloud**")
            fig, ax = plt.subplots()
            ax.imshow(wc_ham, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # 4. Feature Importance from RandomForest
        st.markdown("### 🔑 Top Important Words (from Random Forest)")
        try:
            rf_model = pipeline.named_steps["classifier"]
            vectorizer = pipeline.named_steps["vectorizer"]
            feature_names = vectorizer.get_feature_names_out()
            importances = rf_model.feature_importances_

            feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
            top_features = feat_imp.sort_values(by="importance", ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="importance", y="feature", data=top_features, palette="viridis", ax=ax)
            ax.set_title("Top 20 Important Words for Spam Detection")
            st.pyplot(fig)

        except Exception as e:
            st.warning("⚠️ Could not display feature importances. The classifier may not support it.")
            st.text(str(e))

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("🔧 Developed by **Akinmade Faruq** | 🎓 Final Year Project *(Spam Detection)*")
