"""
Interactive MNIST CNN Web Application
Built with Streamlit and TensorFlow/Keras

This application trains a CNN model on MNIST handwritten digits and provides
an interactive interface for digit prediction, visualization, and model evaluation.
"""

# ===================================================================
# IMPORTS
# ===================================================================

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import io

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ===================================================================
# PAGE CONFIGURATION
# ===================================================================

st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================================================
# CUSTOM CSS STYLING
# ===================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test) - preprocessed data
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to include channel dimension (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return (x_train, y_train), (x_test, y_test)


def build_cnn_model():
    """
    Build and compile the CNN model architecture.
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=(28, 28, 1), name='conv1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        
        # Third Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv3'),
        
        # Flattening and Dense Layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        
        # Output Layer
        layers.Dense(10, activation='softmax', name='output')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


@st.cache_resource
def get_trained_model():
    """
    Load or train the CNN model. Uses caching to avoid retraining.
    
    Returns:
        tuple: (model, test_accuracy, history) - trained model and metrics
    """
    model_path = 'mnist_cnn_model.h5'
    
    # Load preprocessed data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Check if saved model exists
    if os.path.exists(model_path):
        st.info("📂 Loading pre-trained model from disk...")
        model = keras.models.load_model(model_path)
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        st.success(f"✅ Model loaded successfully! Test Accuracy: {test_accuracy*100:.2f}%")
        
        return model, test_accuracy, None
    
    else:
        st.warning("⚠️ No saved model found. Training new model...")
        
        # Build model
        model = build_cnn_model()
        
        # Display model architecture
        with st.expander("🔍 View Model Architecture"):
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom callback to update Streamlit progress
        class StreamlitProgressCallback(keras.callbacks.Callback):
            def __init__(self, epochs):
                self.epochs = epochs
                
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/{self.epochs} - "
                               f"Loss: {logs['loss']:.4f} - "
                               f"Accuracy: {logs['accuracy']:.4f} - "
                               f"Val Loss: {logs['val_loss']:.4f} - "
                               f"Val Accuracy: {logs['val_accuracy']:.4f}")
        
        # Train the model
        st.info("🚀 Starting model training... This may take a few minutes.")
        
        history = model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=10,
            validation_split=0.1,
            verbose=0,
            callbacks=[StreamlitProgressCallback(10)]
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Save the model
        model.save(model_path)
        st.success(f"✅ Model trained and saved! Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return model, test_accuracy, history


def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess an uploaded image for prediction.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (processed_image, display_image) - image ready for prediction and display
    """
    try:
        # Open image using PIL
        image = Image.open(uploaded_file)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Store original for display
        display_image = image.copy()
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert if background is white (MNIST has white digits on black background)
        # Check if the mean pixel value is high (indicating white background)
        if img_array.mean() > 127:
            img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Reshape for model input (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array, display_image
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None


def predict_digit(model, image_array):
    """
    Predict the digit in the image.
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        
    Returns:
        tuple: (predicted_digit, confidence) - prediction and confidence percentage
    """
    predictions = model.predict(image_array, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = predictions[0][predicted_digit] * 100
    
    return predicted_digit, confidence, predictions[0]


def plot_random_predictions(model, x_test, y_test, num_samples=5):
    """
    Plot random test samples with predictions.
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
        num_samples: Number of samples to display
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Select random indices
    random_indices = random.sample(range(len(x_test)), num_samples)
    
    # Get predictions
    predictions = model.predict(x_test[random_indices], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle('CNN Predictions on MNIST Test Set', fontsize=16, fontweight='bold')
    
    # Plot each sample
    for i, (idx, ax) in enumerate(zip(random_indices, axes)):
        image = x_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = predicted_labels[i]
        confidence = predictions[i][pred_label] * 100
        
        # Display image
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        
        # Set title color based on correctness
        color = 'green' if pred_label == true_label else 'red'
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Keras History object
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_prediction_distribution(probabilities):
    """
    Plot the probability distribution across all digits.
    
    Args:
        probabilities: Array of probabilities for each digit
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    digits = np.arange(10)
    colors = ['#1f77b4' if i != np.argmax(probabilities) else '#ff7f0e' 
              for i in range(10)]
    
    ax.bar(digits, probabilities * 100, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Digit', fontsize=12, fontweight='bold')
    ax.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(digits)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ===================================================================
# MAIN APPLICATION
# ===================================================================

def main():
    """
    Main application function.
    """
    
    # Header
    st.markdown('<p class="main-header">🔢 MNIST Digit Recognizer</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning CNN Model for Handwritten Digit Classification</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This application uses a Convolutional Neural Network (CNN) to recognize 
        handwritten digits from the MNIST dataset.
        
        **Features:**
        - 🎯 98%+ accuracy on test data
        - 🖼️ Upload your own digit images
        - 📊 Interactive visualizations
        - 🚀 Real-time predictions
        """)
        
        st.markdown("---")
        
        st.header("⚙️ Model Info")
        st.write("""
        **Architecture:**
        - 3 Convolutional layers
        - 2 MaxPooling layers
        - 2 Dense layers
        - Dropout regularization
        
        **Training:**
        - Dataset: MNIST (60,000 images)
        - Epochs: 10
        - Batch size: 128
        - Optimizer: Adam
        """)
        
        st.markdown("---")
        
        st.header("💡 Tips")
        st.write("""
        For best results when uploading images:
        - Use 28×28 pixel images
        - White digit on black background
        - Center the digit
        - Clear, bold handwriting
        """)
    
    # Load/Train Model
    with st.spinner("🔄 Initializing model..."):
        model, test_accuracy, history = get_trained_model()
    
    # Display Test Accuracy
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(
            label="📊 Model Test Accuracy",
            value=f"{test_accuracy*100:.2f}%",
            delta="Target: >95%" if test_accuracy > 0.95 else None
        )
        
        if test_accuracy > 0.95:
            st.success("✅ Model meets performance requirements!")
        else:
            st.warning("⚠️ Model accuracy below 95% target")
    
    st.markdown("---")
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 Upload & Predict", 
        "🔮 Random Predictions", 
        "📈 Training History",
        "ℹ️ How It Works"
    ])
    
    # ===================================================================
    # TAB 1: UPLOAD & PREDICT
    # ===================================================================
    with tab1:
        st.header("🎨 Upload Your Handwritten Digit")
        st.write("Upload an image of a handwritten digit (0-9) for prediction.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file (PNG, JPG, JPEG)",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of a single handwritten digit"
            )
            
            if uploaded_file is not None:
                # Preprocess image
                processed_image, display_image = preprocess_uploaded_image(uploaded_file)
                
                if processed_image is not None:
                    # Display uploaded image
                    st.subheader("📷 Uploaded Image")
                    st.image(display_image, caption="Your uploaded digit", 
                            use_column_width=True)
                    
                    # Predict button
                    if st.button("🔍 Predict Digit", key="predict_btn"):
                        with st.spinner("Analyzing image..."):
                            predicted_digit, confidence, probabilities = predict_digit(
                                model, processed_image
                            )
                        
                        # Store results in session state
                        st.session_state['prediction'] = predicted_digit
                        st.session_state['confidence'] = confidence
                        st.session_state['probabilities'] = probabilities
        
        with col2:
            # Display prediction results
            if 'prediction' in st.session_state:
                st.subheader("🎯 Prediction Results")
                
                # Main prediction box
                st.markdown(f"""
                <div class="prediction-box">
                    <h1 style="font-size: 4rem; margin: 0; color: #1f77b4;">
                        {st.session_state['prediction']}
                    </h1>
                    <p style="font-size: 1.5rem; margin-top: 1rem; color: #666;">
                        Confidence: {st.session_state['confidence']:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Confidence distribution
                st.subheader("📊 Confidence Distribution")
                fig = plot_prediction_distribution(st.session_state['probabilities'])
                st.pyplot(fig)
                plt.close()
                
                # Detailed probabilities
                with st.expander("📋 View Detailed Probabilities"):
                    prob_df = {
                        'Digit': list(range(10)),
                        'Probability (%)': [f"{p*100:.2f}" for p in st.session_state['probabilities']]
                    }
                    st.table(prob_df)
    
    # ===================================================================
    # TAB 2: RANDOM PREDICTIONS
    # ===================================================================
    with tab2:
        st.header("🔮 Random Test Set Predictions")
        st.write("Visualize model predictions on random samples from the MNIST test set.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎲 Generate Random Predictions", key="random_btn"):
                with st.spinner("Generating predictions..."):
                    # Load test data
                    _, (x_test, y_test) = load_and_preprocess_data()
                    
                    # Generate plot
                    fig = plot_random_predictions(model, x_test, y_test, num_samples=5)
                    st.pyplot(fig)
                    plt.close()
                    
                    st.success("✅ Predictions generated! Click again for new samples.")
    
    # ===================================================================
    # TAB 3: TRAINING HISTORY
    # ===================================================================
    with tab3:
        st.header("📈 Training History & Performance")
        
        if history is not None:
            st.write("Training history for the current model:")
            
            # Plot training history
            fig = plot_training_history(history)
            st.pyplot(fig)
            plt.close()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                final_train_acc = history.history['accuracy'][-1]
                st.metric("Final Training Accuracy", f"{final_train_acc*100:.2f}%")
            
            with col2:
                final_val_acc = history.history['val_accuracy'][-1]
                st.metric("Final Validation Accuracy", f"{final_val_acc*100:.2f}%")
            
            with col3:
                final_train_loss = history.history['loss'][-1]
                st.metric("Final Training Loss", f"{final_train_loss:.4f}")
            
            with col4:
                final_val_loss = history.history['val_loss'][-1]
                st.metric("Final Validation Loss", f"{final_val_loss:.4f}")
            
            # Epoch-by-epoch data
            with st.expander("📊 View Epoch-by-Epoch Metrics"):
                epoch_data = {
                    'Epoch': list(range(1, len(history.history['accuracy']) + 1)),
                    'Train Acc': [f"{acc:.4f}" for acc in history.history['accuracy']],
                    'Val Acc': [f"{acc:.4f}" for acc in history.history['val_accuracy']],
                    'Train Loss': [f"{loss:.4f}" for loss in history.history['loss']],
                    'Val Loss': [f"{loss:.4f}" for loss in history.history['val_loss']]
                }
                st.table(epoch_data)
        else:
            st.info("📝 Training history is only available for newly trained models. "
                   "The current model was loaded from a saved file.")
            
            st.write("**Model Performance:**")
            st.metric("Test Accuracy", f"{test_accuracy*100:.2f}%")
    
    # ===================================================================
    # TAB 4: HOW IT WORKS
    # ===================================================================
    with tab4:
        st.header("ℹ️ How This Application Works")
        
        st.subheader("🧠 The Model")
        st.write("""
        This application uses a **Convolutional Neural Network (CNN)**, a type of deep learning 
        model specifically designed for image recognition tasks. CNNs are particularly effective 
        at learning spatial hierarchies of features from images.
        """)
        
        st.subheader("🏗️ Architecture")
        st.write("""
        The model consists of:
        
        1. **Convolutional Layers**: Extract features like edges, curves, and patterns
        2. **Pooling Layers**: Reduce dimensionality while retaining important information
        3. **Dense Layers**: Combine features to make final predictions
        4. **Dropout**: Prevent overfitting by randomly disabling neurons during training
        """)
        
        st.subheader("📚 Training Process")
        st.write("""
        The model is trained on the **MNIST dataset**, which contains:
        - 60,000 training images
        - 10,000 test images
        - Each image is 28×28 pixels in grayscale
        - Digits range from 0 to 9
        
        Training involves:
        1. **Forward Pass**: Input images → Model → Predictions
        2. **Loss Calculation**: Compare predictions to actual labels
        3. **Backward Pass**: Adjust weights to minimize error
        4. **Iteration**: Repeat for multiple epochs
        """)
        
        st.subheader("🎯 Making Predictions")
        st.write("""
        When you upload an image:
        1. Image is converted to grayscale
        2. Resized to 28×28 pixels
        3. Normalized (pixel values 0-1)
        4. Fed through the trained model
        5. Model outputs probabilities for each digit (0-9)
        6. Highest probability determines the prediction
        """)
        
        st.subheader("📊 Performance Metrics")
        st.write("""
        - **Accuracy**: Percentage of correct predictions
        - **Loss**: Measure of prediction error (lower is better)
        - **Confidence**: Model's certainty in its prediction (0-100%)
        """)
        
        st.subheader("🔬 Technical Details")
        with st.expander("View Technical Specifications"):
            st.code("""
Model Summary:
- Input Shape: (28, 28, 1)
- Conv2D Layer 1: 32 filters, 3×3 kernel
- MaxPooling2D: 2×2
- Conv2D Layer 2: 64 filters, 3×3 kernel
- MaxPooling2D: 2×2
- Conv2D Layer 3: 64 filters, 3×3 kernel
- Flatten Layer
- Dense Layer: 128 neurons, ReLU activation
- Dropout: 50%
- Output Layer: 10 neurons, Softmax activation

Training Configuration:
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 128
- Epochs: 10
- Validation Split: 10%
            """, language='text')
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>🔢 MNIST CNN Digit Recognizer | Deep Learning Project</p>
    </div>
    """, unsafe_allow_html=True)


# ===================================================================
# RUN APPLICATION
# ===================================================================

if __name__ == "__main__":
    main()