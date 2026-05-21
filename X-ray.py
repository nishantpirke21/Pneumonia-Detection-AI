import streamlit as st
import tensorflow as tf
import time

# Page config
st.set_page_config(
    page_title="AI X-Ray Detector",
    page_icon="🧠",
    layout="centered"
)

# 🔥 Advanced CSS Styling
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: -webkit-linear-gradient(#00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-top: 20px;
}

.result-good {
    background: rgba(76, 175, 80, 0.2);
    color: #4CAF50;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.result-bad {
    background: rgba(255, 75, 75, 0.2);
    color: #ff4b4b;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stProgress > div > div > div > div {
    background-color: #00c6ff;
}
</style>
""", unsafe_allow_html=True)

# 🎯 Title
st.markdown(
    '<p class="title">🩺 AI X-Ray Pneumonia Detector</p>',
    unsafe_allow_html=True
)

st.write("### Upload an X-ray image and get instant AI diagnosis")

# Sidebar
st.sidebar.header("⚙️ Settings")
st.sidebar.info("Deep Learning powered medical assistant")

# Settings
img_size = 100
MODEL_PATH = "custom_pre_trained_model_10.h5"

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# 📦 Upload Card
st.markdown('<div class="card">', unsafe_allow_html=True)

file = st.file_uploader(
    "📤 Upload Chest X-Ray Image",
    type=['jpg', 'jpeg', 'png']
)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
if file is not None:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.image(
        file,
        caption="🖼️ Uploaded X-ray",
        use_container_width=True
    )

    # Image preprocessing
    img = tf.keras.preprocessing.image.load_img(
        file,
        target_size=(img_size, img_size)
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, img_size, img_size, 3)

    # Predict button
    if st.button("🔍 Analyze X-ray", key="analyze_btn"):

        # Loading animation
        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        # Prediction
        prediction = model.predict(img_array)

        prob = prediction[0][0]

        # Confidence logic
        if prob > 0.5:
            label = "PNEUMONIA"
            confidence = round(prob * 100, 2)
        else:
            label = "NORMAL"
            confidence = round((1 - prob) * 100, 2)

        # Confidence score
        st.subheader("📊 Confidence Score")
        st.progress(int(confidence))

        # Result display
        if label == "NORMAL":
            st.markdown(
                f'''
                <div class="result-good">
                    ✅ {label}<br>
                    Confidence: {confidence}%
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'''
                <div class="result-bad">
                    ⚠️ {label}<br>
                    Confidence: {confidence}%
                </div>
                ''',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "⚡ Built with Deep Learning | 🧠 AI Project by You"
)