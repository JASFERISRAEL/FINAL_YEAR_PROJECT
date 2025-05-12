# # import json
# # from PIL import Image
# # import io
# # import os
# # import numpy as np
# # import streamlit as st
# # from streamlit import session_state
# # from tensorflow.keras.models import load_model
# # from keras.preprocessing import image as img_preprocessing
# # from tensorflow.keras.applications.efficientnet import preprocess_input
# # import base64


# # session_state = st.session_state
# # if "user_index" not in st.session_state:
# #     st.session_state["user_index"] = 0


# # def signup(json_file_path="data.json"):
# #     st.title("Signup Page")
# #     with st.form("signup_form"):
# #         st.write("Fill in the details below to create an account:")
# #         name = st.text_input("Name:")
# #         email = st.text_input("Email:")
# #         age = st.number_input("Age:", min_value=0, max_value=120)
# #         sex = st.radio("Sex:", ("Male", "Female", "Other"))
# #         password = st.text_input("Password:", type="password")
# #         confirm_password = st.text_input("Confirm Password:", type="password")

# #         if st.form_submit_button("Signup"):
# #             if password == confirm_password:
# #                 user = create_account(name, email, age, sex, password, json_file_path)
# #                 session_state["logged_in"] = True
# #                 session_state["user_info"] = user
# #             else:
# #                 st.error("Passwords do not match. Please try again.")


# # def check_login(username, password, json_file_path="data.json"):
# #     try:
# #         with open(json_file_path, "r") as json_file:
# #             data = json.load(json_file)

# #         for user in data["users"]:
# #             if user["email"] == username and user["password"] == password:
# #                 session_state["logged_in"] = True
# #                 session_state["user_info"] = user
# #                 st.success("Login successful!")
# #                 render_dashboard(user)
# #                 return user

# #         st.error("Invalid credentials. Please try again.")
# #         return None
# #     except Exception as e:
# #         st.error(f"Error checking login: {e}")
# #         return None


# # def predict_retina(image_path, model):
# #     img = img_preprocessing.load_img(image_path, target_size=(224, 224))
# #     img_array = img_preprocessing.img_to_array(img)
# #     img_array = np.expand_dims(img_array, axis=0)
# #     img_array = preprocess_input(img_array)

# #     # Make predictions using the model
# #     predictions = model.predict(img_array)

# #     # Get the predicted label
# #     predicted_label_index = np.argmax(predictions)

# #     # Define class labels
# #     class_labels = {
# #         0: "Healthy",
# #         1: "Mild DR",
# #         2: "Moderate DR",
# #         3: "Proliferate DR",
# #         4: "Severe DR"
# #     }

# #     predicted_label = class_labels[predicted_label_index]

# #     return predicted_label


# # def generate_medical_report(predicted_label):
# #     # Define class labels and corresponding medical information
# #     medical_info = {
# #         "Healthy": {
# #             "report": "Great news! It seems like the patient's eyes are healthy and free from diabetic retinopathy. Regular check-ups are recommended to keep an eye on their eye health.",
# #             "preventative_measures": [
# #                 "Keep up the good work with a healthy lifestyle",
# #                 "Keep blood sugar levels in check",
# #                 "Regular exercise can further boost eye health",
# #             ],
# #             "precautionary_measures": [
# #                 "Stay on top of regular eye exams",
# #                 "Consider scheduling annual comprehensive eye check-ups to monitor any changes",
# #             ],
# #         },
# #         "Mild DR": {
# #             "report": "It looks like there are some early signs of diabetic retinopathy. Nothing to panic about, but it's important to keep a close eye on things and make some lifestyle adjustments.",
# #             "preventative_measures": [
# #                 "Maintain strict control over blood sugar levels",
# #                 "A healthy diet can make a big difference",
# #                 "Regular exercise is key to managing diabetes",
# #             ],
# #             "precautionary_measures": [
# #                 "Plan for more frequent eye check-ups",
# #                 "Consider consulting with an eye specialist to discuss any concerns",
# #             ],
# #         },
# #         "Moderate DR": {
# #             "report": "The patient appears to have moderate diabetic retinopathy. This calls for immediate attention and some lifestyle changes to manage the condition effectively.",
# #             "preventative_measures": [
# #                 "Keep blood sugar levels closely monitored and controlled",
# #                 "A balanced diet is crucial for managing diabetes",
# #                 "Regular exercise can improve circulation and eye health",
# #             ],
# #             "precautionary_measures": [
# #                 "Don't delay regular eye exams",
# #                 "Seek advice from an eye specialist for personalized guidance and treatment options",
# #             ],
# #         },
# #         "Proliferate DR": {
# #             "report": "It seems like the patient is dealing with proliferative diabetic retinopathy. Urgent action is needed to prevent vision loss.",
# #             "preventative_measures": [
# #                 "Maintain strict control over blood sugar levels",
# #                 "Follow a healthy diet to support overall eye health",
# #                 "Regular exercise can help manage diabetes and improve blood flow",
# #             ],
# #             "precautionary_measures": [
# #                 "Seek immediate consultation with an eye specialist",
# #                 "Explore treatment options such as laser therapy or surgery to prevent further complications",
# #             ],
# #         },
# #         "Severe DR": {
# #             "report": "The patient's condition appears to be severe diabetic retinopathy. Immediate medical intervention is critical to prevent blindness.",
# #             "preventative_measures": [
# #                 "Maintain strict control over blood sugar levels",
# #                 "Follow a healthy diet and lifestyle to support overall health",
# #                 "Prompt treatment is essential to preserve vision",
# #             ],
# #             "precautionary_measures": [
# #                 "Seek emergency consultation with an eye specialist",
# #                 "Consider aggressive treatment options such as laser therapy or surgery to halt disease progression",
# #             ],
# #         },
# #     }

# #     # Retrieve medical information based on predicted label
# #     medical_report = medical_info[predicted_label]["report"]
# #     preventative_measures = medical_info[predicted_label]["preventative_measures"]
# #     precautionary_measures = medical_info[predicted_label]["precautionary_measures"]

# #     # Generate conversational medical report with each section in a paragraphic fashion
# #     report = (
# #         "Medical Report:\n\n"
# #         + medical_report
# #         + "\n\nPreventative Measures:\n\n- "
# #         + ",\n- ".join(preventative_measures)
# #         + "\n\nPrecautionary Measures:\n\n- "
# #         + ",\n- ".join(precautionary_measures)
# #     )
# #     precautions = precautionary_measures

# #     return report, precautions


# # def initialize_database(json_file_path="data.json"):
# #     try:
# #         # Check if JSON file exists
# #         if not os.path.exists(json_file_path):
# #             # Create an empty JSON structure
# #             data = {"users": []}
# #             with open(json_file_path, "w") as json_file:
# #                 json.dump(data, json_file)
# #     except Exception as e:
# #         print(f"Error initializing database: {e}")



# # def save_retina_image(image_file, json_file_path="data.json"):
# #     try:
# #         if image_file is None:
# #             st.warning("No file uploaded.")
# #             return

# #         if not session_state["logged_in"] or not session_state["user_info"]:
# #             st.warning("Please log in before uploading images.")
# #             return

# #         # Load user data from JSON file
# #         with open(json_file_path, "r") as json_file:
# #             data = json.load(json_file)

# #         # Find the user's information
# #         for user_info in data["users"]:
# #             if user_info["email"] == session_state["user_info"]["email"]:
# #                 image = Image.open(image_file)

# #                 if image.mode == "RGBA":
# #                     image = image.convert("RGB")

# #                 # Convert image bytes to Base64-encoded string
# #                 image_bytes = io.BytesIO()
# #                 image.save(image_bytes, format="JPEG")
# #                 image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

# #                 # Update the user's information with the Base64-encoded image string
# #                 user_info["retina"] = image_base64

# #                 # Save the updated data to JSON
# #                 with open(json_file_path, "w") as json_file:
# #                     json.dump(data, json_file, indent=4)

# #                 session_state["user_info"]["retina"] = image_base64
# #                 return

# #         st.error("User not found.")
# #     except Exception as e:
# #         st.error(f"Error saving retina image to JSON: {e}")

# # def create_account(name, email, age, sex, password, json_file_path="data.json"):
# #     try:
# #         # Check if the JSON file exists or is empty
# #         if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
# #             data = {"users": []}
# #         else:
# #             with open(json_file_path, "r") as json_file:
# #                 data = json.load(json_file)

# #         # Append new user data to the JSON structure
# #         user_info = {
# #             "name": name,
# #             "email": email,
# #             "age": age,
# #             "sex": sex,
# #             "password": password,
# #             "report": None,
# #             "precautions": None,
# #             "retina":None

# #         }
# #         data["users"].append(user_info)

# #         # Save the updated data to JSON
# #         with open(json_file_path, "w") as json_file:
# #             json.dump(data, json_file, indent=4)

# #         st.success("Account created successfully! You can now login.")
# #         return user_info
# #     except json.JSONDecodeError as e:
# #         st.error(f"Error decoding JSON: {e}")
# #         return None
# #     except Exception as e:
# #         st.error(f"Error creating account: {e}")
# #         return None



# # def login(json_file_path="data.json"):
# #     st.title("Login Page")
# #     username = st.text_input("Username:")
# #     password = st.text_input("Password:", type="password")

# #     login_button = st.button("Login")

# #     if login_button:
# #         user = check_login(username, password, json_file_path)
# #         if user is not None:
# #             session_state["logged_in"] = True
# #             session_state["user_info"] = user
# #         else:
# #             st.error("Invalid credentials. Please try again.")

# # def get_user_info(email, json_file_path="data.json"):
# #     try:
# #         with open(json_file_path, "r") as json_file:
# #             data = json.load(json_file)
# #             for user in data["users"]:
# #                 if user["email"] == email:
# #                     return user
# #         return None
# #     except Exception as e:
# #         st.error(f"Error getting user information: {e}")
# #         return None


# # def render_dashboard(user_info, json_file_path="data.json"):
# #     try:
# #         st.title(f"Welcome to the Dashboard, {user_info['name']}!")
# #         st.subheader("User Information:")
# #         st.write(f"Name: {user_info['name']}")
# #         st.write(f"Sex: {user_info['sex']}")
# #         st.write(f"Age: {user_info['age']}")

# #         # Open the JSON file and check for the 'retina' key
# #         with open(json_file_path, "r") as json_file:
# #             data = json.load(json_file)
# #             for user in data["users"]:
# #                 if user["email"] == user_info["email"]:
# #                     if "retina" in user and user["retina"] is not None:
# #                         image_data = base64.b64decode(user["retina"])
# #                         st.image(Image.open(io.BytesIO(image_data)), caption="Uploaded Retina Image", use_column_width=True)

# #                     if isinstance(user_info["precautions"], list):
# #                         st.subheader("Precautions:")
# #                         for precautopn in user_info["precautions"]:
# #                             st.write(precautopn)                    
# #                     else:
# #                         st.warning("Reminder: Please upload retina images and generate a report.")
# #     except Exception as e:
# #         st.error(f"Error rendering dashboard: {e}")
# # def fetch_precautions(user_info):
# #     return (
# #         user_info["precautions"]
# #         if user_info["precautions"] is not None
# #         else "Please upload retina images and generate a report."
# #     )


# # def main(json_file_path="data.json"):
# #     st.sidebar.title("Diabetic Retinopathy prediction system")
# #     page = st.sidebar.radio(
# #         "Go to",
# #         ("Signup/Login", "Dashboard", "Upload Retina Image", "View Reports"),
# #         key="Diabetic Retinopathy prediction system",
# #     )

# #     if page == "Signup/Login":
# #         st.title("Signup/Login Page")
# #         login_or_signup = st.radio(
# #             "Select an option", ("Login", "Signup"), key="login_signup"
# #         )
# #         if login_or_signup == "Login":
# #             login(json_file_path)
# #         else:
# #             signup(json_file_path)

# #     elif page == "Dashboard":
# #         if session_state.get("logged_in"):
# #             render_dashboard(session_state["user_info"])
# #         else:
# #             st.warning("Please login/signup to view the dashboard.")

# #     elif page == "Upload Retina Image":
# #         if session_state.get("logged_in"):
# #             st.title("Upload Retina Image")
# #             uploaded_image = st.file_uploader(
# #                 "Choose a retina image (JPEG/PNG)", type=["jpg", "jpeg", "png"]
# #             )
# #             if st.button("Upload") and uploaded_image is not None:
# #                 st.image(uploaded_image, use_column_width=True)
# #                 st.success("Retina image uploaded successfully!")
# #                 save_retina_image(uploaded_image, json_file_path)

# #                 model = load_model("models/efficientnet_model.h5")
# #                 condition = predict_retina(uploaded_image, model)
# #                 report, precautions = generate_medical_report(condition)

# #                 # Read the JSON file, update user info, and write back to the file
# #                 with open(json_file_path, "r+") as json_file:
# #                     data = json.load(json_file)
# #                     user_index = next((i for i, user in enumerate(data["users"]) if user["email"] == session_state["user_info"]["email"]), None)
# #                     if user_index is not None:
# #                         user_info = data["users"][user_index]
# #                         user_info["report"] = report
# #                         user_info["precautions"] = precautions
# #                         session_state["user_info"] = user_info
# #                         json_file.seek(0)
# #                         json.dump(data, json_file, indent=4)
# #                         json_file.truncate()
# #                     else:
# #                         st.error("User not found.")
# #                 st.write(report)
# #         else:
# #             st.warning("Please login/signup to upload a retina image.")

# #     elif page == "View Reports":
# #         if session_state.get("logged_in"):
# #             st.title("View Reports")
# #             user_info = get_user_info(session_state["user_info"]["email"], json_file_path)
# #             if user_info is not None:
# #                 if user_info["report"] is not None:
# #                     st.subheader("Retina Report:")
# #                     st.write(user_info["report"])
# #                 else:
# #                     st.warning("No reports available.")
# #             else:
# #                 st.warning("User information not found.")
# #         else:
# #             st.warning("Please login/signup to view reports.")



# # if __name__ == "__main__":
# #     initialize_database()
# #     main()
# import os
# import io
# import json
# import sqlite3
# import secrets
# import numpy as np
# from PIL import Image
# from datetime import datetime
# from collections import Counter
# from werkzeug.utils import secure_filename
# from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify

# # ML imports
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image as img_preprocessing
# from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
# from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
# from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = secrets.token_hex(16)

# # Configure upload folder
# UPLOAD_FOLDER = 'uploads'
# TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, 'temp')
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # Make sure upload directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(TEMP_FOLDER, exist_ok=True)
# os.makedirs('models', exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# # Database initialization
# def init_db():
#     conn = sqlite3.connect('retinopathy.db')
#     c = conn.cursor()
#     c.execute('''
#     CREATE TABLE IF NOT EXISTS users (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT NOT NULL,
#         email TEXT UNIQUE NOT NULL,
#         password TEXT NOT NULL
#     )
#     ''')
    
#     c.execute('''
#     CREATE TABLE IF NOT EXISTS reports (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         user_id INTEGER,
#         image_path TEXT,
#         condition TEXT,
#         confidence REAL,
#         cost TEXT,
#         screening_time TEXT,
#         report_text TEXT,
#         patient_name TEXT,
#         patient_age TEXT,
#         patient_sex TEXT,
#         patient_contact TEXT,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#         FOREIGN KEY (user_id) REFERENCES users (id)
#     )
#     ''')
    
#     conn.commit()
#     conn.close()

# # Initialize database on startup
# init_db()

# # Helper Functions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def create_user(name, email, password):
#     try:
#         conn = sqlite3.connect('retinopathy.db')
#         c = conn.cursor()
#         c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
#                 (name, email, password))
#         conn.commit()
#         user_id = c.lastrowid
#         conn.close()
#         return user_id
#     except sqlite3.IntegrityError:
#         return None
#     except Exception as e:
#         print(f"Error creating user: {e}")
#         return None

# def check_login(email, password):
#     try:
#         conn = sqlite3.connect('retinopathy.db')
#         c = conn.cursor()
#         c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
#         user = c.fetchone()
#         conn.close()
        
#         if user:
#             return {
#                 "id": user[0],
#                 "name": user[1],
#                 "email": user[2]
#             }
#         return None
#     except Exception as e:
#         print(f"Error checking login: {e}")
#         return None

# def save_image(file, user_id):
#     if file is None:
#         return None
    
#     # Create user directory if it doesn't exist
#     user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
#     os.makedirs(user_dir, exist_ok=True)
    
#     # Generate unique filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{timestamp}_{secure_filename(file.filename)}"
#     file_path = os.path.join(user_dir, filename)
    
#     # Save the file
#     img = Image.open(file)
#     if img.mode == "RGBA":
#         img = img.convert("RGB")
#     img.save(file_path)
    
#     return file_path

# def preprocess_image(image_path, model_type, target_size=(224, 224)):
#     """Preprocess an image based on the model type."""
#     img = img_preprocessing.load_img(image_path, target_size=target_size)
#     img_array = img_preprocessing.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
    
#     if model_type == "efficientnet":
#         return efficientnet_preprocess(img_array)
#     elif model_type == "resnet":
#         return resnet_preprocess(img_array)
#     elif model_type == "densenet":
#         return densenet_preprocess(img_array)
#     else:
#         return img_array  # Default preprocessing

# def predict_with_model(image_path, model_type):
#     """Make predictions using a specific model type."""
#     try:
#         model_path = f"models/{model_type}_model.h5"
#         model = load_model(model_path)
        
#         # Preprocess based on model type
#         target_size = (224, 224)  # Default for most models
        
#         img_array = preprocess_image(image_path, model_type, target_size)
        
#         # Make predictions
#         predictions = model.predict(img_array)
#         predicted_label_index = np.argmax(predictions)
#         confidence = float(predictions[0][predicted_label_index]) * 100
        
#         # Define class labels
#         class_labels = {
#             0: "Healthy",
#             1: "Mild NPDR",
#             2: "Moderate DR",
#             3: "Proliferate DR",
#             4: "Severe DR"
#         }
        
#         predicted_label = class_labels.get(predicted_label_index, "Unknown")
        
#         return {
#             "condition": predicted_label,
#             "confidence": confidence,
#             "label_index": predicted_label_index
#         }
#     except Exception as e:
#         print(f"Error in {model_type} prediction: {e}")
#         return None

# def predict_retina(image_path):
#     """Predict retina condition using three models and taking the mode."""
#     try:
#         # Make predictions with each model
#         efficientnet_result = predict_with_model(image_path, "efficientnet")
#         resnet_result = predict_with_model(image_path, "resnet")
#         densenet_result = predict_with_model(image_path, "densenet")
        
#         # Debug logging
#         print(f"EfficientNet result: {efficientnet_result}")
#         print(f"ResNet result: {resnet_result}")
#         print(f"DenseNet result: {densenet_result}")
        
#         # Collect results - make sure we have at least one valid result
#         results = []
#         if efficientnet_result:
#             results.append(efficientnet_result)
#         if resnet_result:
#             results.append(resnet_result)
#         if densenet_result:
#             results.append(densenet_result)
        
#         if not results:
#             print("No valid predictions from any model")
#             return None
        
#         # Calculate mode of predictions
#         predictions = [r["condition"] for r in results]
#         prediction_count = Counter(predictions)
#         final_condition = prediction_count.most_common(1)[0][0]
        
#         # Get the confidence from the model that predicted this condition
#         confidence_values = [r["confidence"] for r in results if r["condition"] == final_condition]
#         final_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
#         # Calculate agreement percentage
#         agreement = (prediction_count[final_condition] / len(results)) * 100
        
#         final_result = {
#             "condition": final_condition,
#             "confidence": final_confidence,
#             "agreement": agreement,
#             "model_results": {
#                 "efficientnet": efficientnet_result["condition"] if efficientnet_result else "Failed",
#                 "resnet": resnet_result["condition"] if resnet_result else "Failed",
#                 "densenet": densenet_result["condition"] if densenet_result else "Failed"
#             }
#         }
        
#         return final_result
#     except Exception as e:
#         import traceback
#         print(f"Error in ensemble prediction: {e}")
#         print(traceback.format_exc())
#         return None

# def get_cost_estimation(condition):
#     # Set cost ranges based on condition severity in Indian Rupees (₹)
#     cost_map = {
#         "Healthy": "₹500",
#         "Mild NPDR": "₹1,200",
#         "Moderate DR": "₹2,000",
#         "Proliferate DR": "₹3,500",
#         "Severe DR": "₹5,000",
#         "Unknown": "₹1,000"
#     }
#     return cost_map.get(condition, "₹1,000")

# def get_screening_recommendation(condition):
#     # Set screening times based on condition severity
#     screening_map = {
#         "Healthy": "30 minutes - Annual checkup",
#         "Mild NPDR": "45 minutes - Every 6 months",
#         "Moderate DR": "60 minutes - Every 3-4 months",
#         "Proliferate DR": "90 minutes - Every 1-2 months",
#         "Severe DR": "120 minutes - Immediate referral",
#         "Unknown": "60 minutes - Consult your doctor"
#     }
#     return screening_map.get(condition, "60 minutes - Consult your doctor")

# def get_prevention_tips(condition):
#     tips_map = {
#         "Healthy": [
#             "Regular blood sugar monitoring",
#             "Maintain a healthy diet",
#             "Exercise regularly",
#             "Annual eye examinations"
#         ],
#         "Mild NPDR": [
#             "Regular blood sugar monitoring",
#             "Maintain a healthy diet",
#             "Exercise regularly",
#             "Schedule regular check-ups"
#         ],
#         "Moderate DR": [
#             "Strict blood sugar control",
#             "Blood pressure management",
#             "Regular exercise",
#             "Follow-up eye examination in 3-4 months"
#         ],
#         "Proliferate DR": [
#             "Immediate ophthalmologist consultation",
#             "Strict glycemic control",
#             "Aggressive management of all risk factors",
#             "Monthly follow-up examinations"
#         ],
#         "Severe DR": [
#             "Seek emergency consultation with an eye specialist",
#             "Consider aggressive treatment options",
#             "Strict blood sugar control",
#             "Prompt treatment is essential to preserve vision"
#         ],
#         "Unknown": [
#             "Consult your doctor", 
#             "Follow medical advice", 
#             "Regular check-ups"
#         ]
#     }
#     return tips_map.get(condition, ["Consult your doctor", "Follow medical advice", "Regular check-ups"])

# def generate_report_text(condition):
#     report_map = {
#         "Healthy": "Great news! It seems like the patient's eyes are healthy and free from diabetic retinopathy. Regular check-ups are recommended to keep an eye on their eye health.",
#         "Mild NPDR": "It looks like there are some early signs of diabetic retinopathy. Nothing to panic about, but it's important to keep a close eye on things and make some lifestyle adjustments.",
#         "Moderate DR": "The patient appears to have moderate diabetic retinopathy. This calls for immediate attention and some lifestyle changes to manage the condition effectively.",
#         "Proliferate DR": "It seems like the patient is dealing with proliferative diabetic retinopathy. Urgent action is needed to prevent vision loss.",
#         "Severe DR": "The patient's condition appears to be severe diabetic retinopathy. Immediate medical intervention is critical to prevent blindness.",
#         "Unknown": "The analysis was inconclusive. Please consult with a specialist for further evaluation."
#     }
#     return report_map.get(condition, "Please consult with a specialist for detailed analysis.")

# def generate_text_report(patient_details, condition, confidence, cost, screening_time, prevention_tips, model_results, analysis_time):
#     """Generate a text-based report."""
    
#     # Creating the text report with formatting
#     report = []
#     report.append("=" * 50)
#     report.append("DIABETIC RETINOPATHY DETECTION REPORT")
#     report.append("=" * 50)
#     report.append("")
    
#     # Patient information
#     report.append("PATIENT INFORMATION:")
#     report.append("-" * 30)
#     report.append(f"Patient Name: {patient_details['name']}")
#     report.append(f"Age: {patient_details['age']}")
#     report.append(f"Sex: {patient_details['sex']}")
#     report.append(f"Contact: {patient_details['contact']}")
#     report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     report.append("")
    
#     # Detection Results
#     report.append("DETECTION RESULTS:")
#     report.append("-" * 30)
#     report.append(f"Condition: {condition}")
#     report.append(f"Confidence: {confidence:.1f}%")
#     report.append(f"Estimated Cost: {cost}")
#     report.append(f"Screening Time: {screening_time}")
#     report.append("")
    
#     # Model Results
#     if model_results:
#         report.append("MODEL PREDICTIONS:")
#         report.append("-" * 30)
#         for model, result in model_results.items():
#             report.append(f"{model.capitalize()} Model: {result}")
#         report.append("")
    
#     # Medical Report
#     report.append("MEDICAL REPORT:")
#     report.append("-" * 30)
#     report.append(generate_report_text(condition))
#     report.append("")
    
#     # Prevention Tips
#     report.append("PREVENTION TIPS:")
#     report.append("-" * 30)
#     for i, tip in enumerate(prevention_tips, 1):
#         report.append(f"{i}. {tip}")
#     report.append("")
    
#     # Footer
#     report.append("=" * 50)
#     report.append("Report generated by Diabetic Retinopathy Detection System")
#     report.append(f"Analysis Time: {analysis_time}")
#     report.append("=" * 50)
    
#     # Join all lines with newlines
#     return "\n".join(report)

# def save_report_to_db(user_id, image_path, condition, confidence, cost, screening_time, 
#                      patient_name, patient_age, patient_sex, patient_contact, prevention_tips, 
#                      model_results, analysis_time):
#     try:
#         # Generate the text report
#         patient_details = {
#             "name": patient_name,
#             "age": patient_age,
#             "sex": patient_sex,
#             "contact": patient_contact
#         }
        
#         report_text = generate_text_report(
#             patient_details, 
#             condition, 
#             confidence, 
#             cost, 
#             screening_time, 
#             prevention_tips, 
#             model_results,
#             analysis_time
#         )
        
#         # Connect to the database
#         conn = sqlite3.connect('retinopathy.db')
#         c = conn.cursor()
        
#         # Insert the report
#         c.execute('''
#         INSERT INTO reports (
#             user_id, 
#             image_path, 
#             condition, 
#             confidence, 
#             cost, 
#             screening_time, 
#             report_text,
#             patient_name,
#             patient_age,
#             patient_sex,
#             patient_contact,
#             created_at
#         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         ''', (
#             user_id, 
#             image_path, 
#             condition, 
#             confidence, 
#             cost, 
#             screening_time, 
#             report_text,
#             patient_name,
#             patient_age,
#             patient_sex,
#             patient_contact,
#             datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         ))
        
#         conn.commit()
#         report_id = c.lastrowid
#         conn.close()
        
#         return report_id
#     except Exception as e:
#         print(f"Error saving report: {e}")
#         return None

# def get_last_report(user_id):
#     try:
#         conn = sqlite3.connect('retinopathy.db')
#         conn.row_factory = sqlite3.Row  # Use Row factory to access columns by name
#         c = conn.cursor()
#         c.execute('''
#         SELECT * FROM reports 
#         WHERE user_id = ? 
#         ORDER BY created_at DESC 
#         LIMIT 1
#         ''', (user_id,))
        
#         report = c.fetchone()
#         conn.close()
        
#         if report:
#             # Convert sqlite3.Row to dictionary
#             report_dict = dict(report)
#             return report_dict
        
#         return None
#     except Exception as e:
#         print(f"Error retrieving report: {e}")
#         return None

# def get_report_by_id(report_id):
#     try:
#         conn = sqlite3.connect('retinopathy.db')
#         conn.row_factory = sqlite3.Row  # Use Row factory to access columns by name
#         c = conn.cursor()
#         c.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
        
#         report = c.fetchone()
#         conn.close()
        
#         if report:
#             # Convert sqlite3.Row to dictionary
#             report_dict = dict(report)
#             return report_dict
        
#         return None
#     except Exception as e:
#         print(f"Error retrieving report: {e}")
#         return None

# def get_user_reports(user_id):
#     try:
#         conn = sqlite3.connect('retinopathy.db')
#         conn.row_factory = sqlite3.Row
#         c = conn.cursor()
#         c.execute('''
#         SELECT id, condition, confidence, patient_name, created_at
#         FROM reports
#         WHERE user_id = ?
#         ORDER BY created_at DESC
#         ''', (user_id,))
        
#         reports = [dict(row) for row in c.fetchall()]
#         conn.close()
        
#         return reports
#     except Exception as e:
#         print(f"Error retrieving user reports: {e}")
#         return []

# # Routes
# @app.route('/')
# def index():
#     if 'user_id' in session:
#         return redirect(url_for('dashboard'))
#     return render_template('index.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
        
#         user = check_login(email, password)
#         if user:
#             # Store user info in session
#             session['user_id'] = user['id']
#             session['user_name'] = user['name']
#             session['user_email'] = user['email']
            
#             flash('Login successful!', 'success')
#             return redirect(url_for('dashboard'))
#         else:
#             flash('Invalid credentials. Please try again.', 'error')
    
#     return render_template('login.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         password = request.form['password']
#         confirm_password = request.form['confirm_password']
        
#         if password != confirm_password:
#             flash('Passwords do not match!', 'error')
#         elif not name or not email or not password:
#             flash('Please fill in all required fields.', 'error')
#         else:
#             user_id = create_user(name, email, password)
#             if user_id:
#                 flash('Account created successfully! You can now login.', 'success')
#                 return redirect(url_for('login'))
#             else:
#                 flash('Email already exists or there was an error creating your account.', 'error')
    
#     return render_template('register.html')

# @app.route('/logout')
# def logout():
#     # Clear the session
#     session.clear()
#     flash('You have been logged out.', 'info')
#     return redirect(url_for('index'))

# @app.route('/dashboard')
# def dashboard():
#     if 'user_id' not in session:
#         flash('Please log in to access the dashboard.', 'error')
#         return redirect(url_for('login'))
    
#     # Get all reports for the user
#     reports = get_user_reports(session['user_id'])
    
#     return render_template('dashboard.html', reports=reports)

# @app.route('/analysis', methods=['GET', 'POST'])
# def analysis():
#     if 'user_id' not in session:
#         flash('Please log in to access this page.', 'error')
#         return redirect(url_for('login'))
    
#     if request.method == 'POST':
#         # Check if the post request has the file part
#         if 'retina_image' not in request.files:
#             flash('No file part', 'error')
#             return redirect(request.url)
        
#         file = request.files['retina_image']
        
#         # If user does not select file, browser also submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file', 'error')
#             return redirect(request.url)
        
#         if file and allowed_file(file.filename):
#             # Save the uploaded file
#             image_path = save_image(file, session['user_id'])
            
#             if image_path:
#                 # Predict retina condition
#                 result = predict_retina(image_path)
                
#                 if result:
#                     # Extract results
#                     condition = result["condition"]
#                     confidence = result["confidence"]
#                     agreement = result.get("agreement", 0)
#                     model_results = result.get("model_results", {})
                    
#                     # Get additional info
#                     cost = get_cost_estimation(condition)
#                     screening_time = get_screening_recommendation(condition)
#                     prevention_tips = get_prevention_tips(condition)
#                     analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
#                     # Store results in session
#                     session['analysis_results'] = {
#                         "condition": condition,
#                         "confidence": confidence,
#                         "agreement": agreement,
#                         "model_results": model_results,
#                         "cost": cost,
#                         "screening_time": screening_time,
#                         "prevention_tips": prevention_tips,
#                         "image_path": image_path,
#                         "analysis_time": analysis_time
#                     }
                    
#                     flash('Analysis completed successfully!', 'success')
#                     return redirect(url_for('analysis_results'))
#                 else:
#                     flash('Error analyzing the image. Please try again.', 'error')
#             else:
#                 flash('Error saving the uploaded image.', 'error')
#         else:
#             flash('Invalid file format. Please upload a JPG, JPEG, or PNG image.', 'error')
    
#     return render_template('analysis.html')

# @app.route('/analysis/results')
# def analysis_results():
#     if 'user_id' not in session:
#         flash('Please log in to access this page.', 'error')
#         return redirect(url_for('login'))
    
#     if 'analysis_results' not in session:
#         flash('No analysis results available. Please analyze an image first.', 'error')
#         return redirect(url_for('analysis'))
    
#     # Get results from session
#     results = session['analysis_results']
    
#     # Get relative path for displaying the image
#     image_filename = os.path.basename(results['image_path'])
#     image_dir = os.path.dirname(results['image_path']).replace(app.config['UPLOAD_FOLDER'], 'uploads')
#     image_url = f"/{image_dir}/{image_filename}"
    
#     return render_template('analysis_results.html', 
#                           results=results, 
#                           image_url=image_url)

# @app.route('/report', methods=['GET', 'POST'])
# def report():
#     if 'user_id' not in session:
#         flash('Please log in to access this page.', 'error')
#         return redirect(url_for('login'))
    
#     if 'analysis_results' not in session:
#         flash('No analysis results available. Please analyze an image first.', 'error')
#         return redirect(url_for('analysis'))
    
#     results = session['analysis_results']
    
#     if request.method == 'POST':
#         # Get patient details from form
#         patient_name = request.form['patient_name']
#         patient_age = request.form['patient_age']
#         patient_sex = request.form['patient_sex']
#         patient_contact = request.form['patient_contact']
        
#         if not patient_name or not patient_age:
#             flash('Please provide at least the patient\'s name and age.', 'error')
#         else:
#             # Save report to database
#             report_id = save_report_to_db(
#                 session['user_id'],
#                 results['image_path'],
#                 results['condition'],
#                 results['confidence'],
#                 results['cost'],
#                 results['screening_time'],
#                 patient_name,
#                 patient_age,
#                 patient_sex,
#                 patient_contact,
#                 results['prevention_tips'],
#                 results['model_results'],
#                 results['analysis_time']
#             )
            
#             if report_id:
#                 session['report_id'] = report_id
#                 flash(f'Report #{report_id} generated and saved successfully!', 'success')
#                 return redirect(url_for('view_report', report_id=report_id))
#             else:
#                 flash('Failed to save report. Please try again.', 'error')
    
#     # Get last report for display
#     last_report = get_last_report(session['user_id'])
    
#     return render_template('report.html', 
#                           results=results, 
#                           last_report=last_report)

# @app.route('/report/<int:report_id>')
# def view_report(report_id):
#     if 'user_id' not in session:
#         flash('Please log in to access this page.', 'error')
#         return redirect(url_for('login'))
    
#     # Get the report
#     report = get_report_by_id(report_id)
    
#     if report and report['user_id'] == session['user_id']:
#         # Get relative path for displaying the image
#         image_filename = os.path.basename(report['image_path'])
#         image_dir = os.path.dirname(report['image_path']).replace(app.config['UPLOAD_FOLDER'], 'uploads')
#         image_url = f"/{image_dir}/{image_filename}"
        
#         return render_template('view_report.html', report=report, image_url=image_url)
#     else:
#         flash('Report not found or you do not have permission to view it.', 'error')
#         return redirect(url_for('dashboard'))

# @app.route('/download/report/<int:report_id>')
# def download_report(report_id):
#     if 'user_id' not in session:
#         flash('Please log in to access this page.', 'error')
#         return redirect(url_for('login'))
    
#     # Get the report
#     report = get_report_by_id(report_id)
    
#     if report and report['user_id'] == session['user_id']:
#         # Create file-like object in memory
#         report_io = io.BytesIO()
#         report_io.write(report['report_text'].encode('utf-8'))
#         report_io.seek(0)
        
#         return send_file(
#             report_io,
#             mimetype='text/plain',
#             as_attachment=True,
#             download_name=f"Retinopathy_Report_{report_id}.txt"
#         )
#     else:
#         flash('Report not found or you do not have permission to download it.', 'error')
#         return redirect(url_for('dashboard'))

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import sqlite3
import os
import tempfile
import base64
from PIL import Image
from datetime import datetime
import numpy as np
import io
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from keras.preprocessing import image as img_preprocessing
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from collections import Counter
app = Flask(__name__)
app.secret_key = 'diabetic_retinopathy_key'

# Create folders if they don't exist
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("temp_uploads"):
    os.makedirs("temp_uploads")

# Initialize database
def init_db():
    conn = sqlite3.connect('retinopathy.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_path TEXT,
        condition TEXT,
        confidence REAL,
        cost TEXT,
        screening_time TEXT,
        report_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# User functions
def create_user(name, email, password):
    try:
        conn = sqlite3.connect('retinopathy.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, password))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

def check_login(email, password):
    try:
        conn = sqlite3.connect('retinopathy.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            return {
                "id": user[0],
                "name": user[1],
                "email": user[2]
            }
        return None
    except Exception as e:
        print(f"Error checking login: {e}")
        return None

# Image and prediction functions
def save_image(uploaded_file, user_id):
    if uploaded_file is None:
        return None
    
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join("temp_uploads", str(user_id))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{secure_filename(uploaded_file.filename)}"
    file_path = os.path.join(temp_dir, filename)
    
    # Save the file
    img = Image.open(uploaded_file)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(file_path)
    
    return file_path

# Simplified function to simulate prediction
def preprocess_image(image_path, model_type, target_size=(224, 224)):
    """Preprocess an image based on the model type."""
    img = img_preprocessing.load_img(image_path, target_size=target_size)
    img_array = img_preprocessing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == "efficientnet":
        return efficientnet_preprocess(img_array)
    elif model_type == "resnet":
        return resnet_preprocess(img_array)
    elif model_type == "densenet":
        return densenet_preprocess(img_array)
    else:
        return img_array  # Default preprocessing

def predict_with_model(image_path, model_type):
    """Make predictions using a specific model type."""
    try:
        model_path = f"models/{model_type}_model.h5"
        model = load_model(model_path)
        
        # Preprocess based on model type
        target_size = (224, 224)  # Default for most models
        
        img_array = preprocess_image(image_path, model_type, target_size)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_label_index = np.argmax(predictions)
        confidence = float(predictions[0][predicted_label_index]) * 100
        
        # Define class labels
        class_labels = {
            0: "Healthy",
            1: "Mild NPDR",
            2: "Moderate DR",
            3: "Proliferate DR",
            4: "Severe DR"
        }
        
        predicted_label = class_labels.get(predicted_label_index, "Unknown")
        
        return {
            "condition": predicted_label,
            "confidence": confidence,
            "label_index": predicted_label_index
        }
    except Exception as e:

        return None

def predict_retina(image_path):
    """Predict retina condition using three models with EfficientNet having higher weight."""
    try:
        # Make predictions with each model
        efficientnet_result = predict_with_model(image_path, "efficientnet")
        resnet_result = predict_with_model(image_path, "resnet")
        densenet_result = predict_with_model(image_path, "densenet")
        
        # Debug logging
        print(f"EfficientNet result: {efficientnet_result}")
        print(f"ResNet result: {resnet_result}")
        print(f"DenseNet result: {densenet_result}")
        
        # Collect results - make sure we have at least one valid result
        results = []
        
        # Add EfficientNet result with higher weight (3 times)
        if efficientnet_result:
            for _ in range(3):  # Add three times to increase weight
                results.append(efficientnet_result)
        
        # Add other results with normal weight (1 time each)
        if resnet_result:
            results.append(resnet_result)
        if densenet_result:
            results.append(densenet_result)
        
        if not results:
            print("No valid predictions from any model")
            return None
        
        # Calculate weighted mode of predictions
        predictions = [r["condition"] for r in results]
        prediction_count = Counter(predictions)
        final_condition = prediction_count.most_common(1)[0][0]
        
        # Get the confidence from the models that predicted this condition
        confidence_values = [r["confidence"] for r in results if r["condition"] == final_condition]
        final_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
        # Calculate agreement percentage (considering weights)
        agreement = (prediction_count[final_condition] / len(results)) * 100
        
        final_result = {
            "condition": final_condition,
            "confidence": final_confidence,
            "agreement": agreement,
            "model_results": {
                "efficientnet": efficientnet_result["condition"] if efficientnet_result else "Failed",
                "resnet": resnet_result["condition"] if resnet_result else "Failed",
                "densenet": densenet_result["condition"] if densenet_result else "Failed"
            }
        }
        
        # Debug - print the final result
        print(f"Final result: {final_result}")
        
        return final_result
    except Exception as e:
        import traceback
        print(f"Error in ensemble prediction: {e}")
        print(traceback.format_exc())
        return None

def get_cost_estimation(condition):
    # Set cost ranges based on condition severity in Indian Rupees (₹)
    cost_map = {
        "Healthy": "₹500",
        "Mild NPDR": "₹1,200",
        "Moderate DR": "₹2,000",
        "Proliferate DR": "₹3,500",
        "Severe DR": "₹5,000",
        "Unknown": "₹1,000"
    }
    return cost_map.get(condition, "₹1,000")

def get_screening_recommendation(condition):
    # Set screening times based on condition severity
    screening_map = {
        "Healthy": "30 minutes - Annual checkup",
        "Mild NPDR": "45 minutes - Every 6 months",
        "Moderate DR": "60 minutes - Every 3-4 months",
        "Proliferate DR": "90 minutes - Every 1-2 months",
        "Severe DR": "120 minutes - Immediate referral",
        "Unknown": "60 minutes - Consult your doctor"
    }
    return screening_map.get(condition, "60 minutes - Consult your doctor")

def get_prevention_tips(condition):
    tips_map = {
        "Healthy": [
            "Regular blood sugar monitoring",
            "Maintain a healthy diet",
            "Exercise regularly",
            "Annual eye examinations"
        ],
        "Mild NPDR": [
            "Regular blood sugar monitoring",
            "Maintain a healthy diet",
            "Exercise regularly",
            "Schedule regular check-ups"
        ],
        "Moderate DR": [
            "Strict blood sugar control",
            "Blood pressure management",
            "Regular exercise",
            "Follow-up eye examination in 3-4 months"
        ],
        "Proliferate DR": [
            "Immediate ophthalmologist consultation",
            "Strict glycemic control",
            "Aggressive management of all risk factors",
            "Monthly follow-up examinations"
        ],
        "Severe DR": [
            "Seek emergency consultation with an eye specialist",
            "Consider aggressive treatment options",
            "Strict blood sugar control",
            "Prompt treatment is essential to preserve vision"
        ],
        "Unknown": [
            "Consult your doctor", 
            "Follow medical advice", 
            "Regular check-ups"
        ]
    }
    return tips_map.get(condition, ["Consult your doctor", "Follow medical advice", "Regular check-ups"])

def generate_report_text(condition):
    report_map = {
        "Healthy": "Great news! It seems like the patient's eyes are healthy and free from diabetic retinopathy. Regular check-ups are recommended to keep an eye on their eye health.",
        "Mild NPDR": "It looks like there are some early signs of diabetic retinopathy. Nothing to panic about, but it's important to keep a close eye on things and make some lifestyle adjustments.",
        "Moderate DR": "The patient appears to have moderate diabetic retinopathy. This calls for immediate attention and some lifestyle changes to manage the condition effectively.",
        "Proliferate DR": "It seems like the patient is dealing with proliferative diabetic retinopathy. Urgent action is needed to prevent vision loss.",
        "Severe DR": "The patient's condition appears to be severe diabetic retinopathy. Immediate medical intervention is critical to prevent blindness.",
        "Unknown": "The analysis was inconclusive. Please consult with a specialist for further evaluation."
    }
    return report_map.get(condition, "Please consult with a specialist for detailed analysis.")

def generate_text_report(patient_details, condition, confidence, cost, screening_time, prevention_tips, model_results, analysis_time):
    """Generate a text-based report"""
    
    # Creating the text report with formatting
    report = []
    report.append("=" * 50)
    report.append("DIABETIC RETINOPATHY DETECTION REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Patient information
    report.append("PATIENT INFORMATION:")
    report.append("-" * 30)
    report.append(f"Patient Name: {patient_details['name']}")
    report.append(f"Age: {patient_details['age']}")
    report.append(f"Sex: {patient_details['sex']}")
    report.append(f"Contact: {patient_details['contact']}")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Detection Results
    report.append("DETECTION RESULTS:")
    report.append("-" * 30)
    report.append(f"Condition: {condition}")
    report.append(f"Confidence: {confidence:.1f}%")
    report.append(f"Estimated Cost: {cost}")
    report.append(f"Screening Time: {screening_time}")
    report.append("")
    
    # Model Results
    if model_results:
        report.append("MODEL PREDICTIONS:")
        report.append("-" * 30)
        for model, result in model_results.items():
            report.append(f"{model.capitalize()} Model: {result}")
        report.append("")
    
    # Medical Report
    report.append("MEDICAL REPORT:")
    report.append("-" * 30)
    report.append(generate_report_text(condition))
    report.append("")
    
    # Prevention Tips
    report.append("PREVENTION TIPS:")
    report.append("-" * 30)
    for i, tip in enumerate(prevention_tips, 1):
        report.append(f"{i}. {tip}")
    report.append("")
    
    # Footer
    report.append("=" * 50)
    report.append("Report generated by Diabetic Retinopathy Detection System")
    report.append(f"Analysis Time: {analysis_time}")
    report.append("=" * 50)
    
    # Join all lines with newlines
    return "\n".join(report)

# Routes
@app.route('/')
def index():
    if 'logged_in' in session and session['logged_in']:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = check_login(email, password)
        if user:
            session['logged_in'] = True
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user['email']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
        elif not name or not email or not password:
            flash('Please fill in all required fields', 'error')
        else:
            user_id = create_user(name, email, password)
            if user_id:
                flash('Account created successfully! You can now login', 'success')
                return redirect(url_for('login'))
            else:
                flash('Email already exists or there was an error creating your account', 'error')
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', username=session['user_name'])

@app.route('/upload', methods=['POST'])
def upload():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    
    uploaded_file = request.files.get('retina_image')
    if not uploaded_file:
        flash('No file selected', 'error')
        return redirect(url_for('dashboard'))
    
    # Save the image
    image_path = save_image(uploaded_file, session['user_id'])
    
    # Store image path in session for analysis
    session['image_path'] = image_path
    
    return redirect(url_for('analyze'))

@app.route('/analyze')
def analyze():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    
    if 'image_path' not in session or not session['image_path']:
        flash('Please upload an image first', 'error')
        return redirect(url_for('dashboard'))
    
    # Get the image path from session
    image_path = session['image_path']
    
    # Predict retina condition
    result = predict_retina(image_path)
    
    # Generate additional information
    condition = result["condition"]
    confidence = result["confidence"]
    cost = get_cost_estimation(condition)
    screening_time = get_screening_recommendation(condition)
    prevention_tips = get_prevention_tips(condition)
    
    # Store results in session for report generation
    session['analysis_results'] = {
        "condition": condition,
        "confidence": confidence,
        "model_results": result.get("model_results", {}),
        "cost": cost,
        "screening_time": screening_time,
        "prevention_tips": prevention_tips,
        "image_path": image_path,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Display the uploaded image
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    
    return render_template(
        'results.html',
        img_data=img_data,
        condition=condition,
        confidence=confidence,
        cost=cost,
        screening_time=screening_time,
        prevention_tips=prevention_tips
    )

@app.route('/report')
def report():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    
    if 'analysis_results' not in session or not session['analysis_results']:
        flash('No analysis results available', 'error')
        return redirect(url_for('dashboard'))
    
    results = session['analysis_results']
    
    return render_template('report.html', results=results)

@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    
    if 'analysis_results' not in session or not session['analysis_results']:
        flash('No analysis results available', 'error')
        return redirect(url_for('dashboard'))
    
    # Get report type (txt or pdf)
    report_type = request.form.get('report_type', 'txt')
    
    # Get patient details from form
    patient_details = {
        "name": request.form.get('patient_name', ''),
        "age": request.form.get('patient_age', ''),
        "sex": request.form.get('patient_sex', 'Male'),
        "contact": request.form.get('patient_contact', '')
    }
    
    # Get analysis results from session
    results = session['analysis_results']
    
    # Generate text report
    text_report = generate_text_report(
        patient_details,
        results['condition'],
        results['confidence'],
        results['cost'],
        results['screening_time'],
        results['prevention_tips'],
        results['model_results'],
        results['analysis_time']
    )
    
    # For simplicity, we'll only implement text report for now
    # In a real app, you would implement PDF generation as well
    
    # Create a temporary file for the report
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(text_report.encode())
        report_path = tmp_file.name
    
    # Send the file to the user for download
    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"Retinopathy_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mimetype='text/plain'
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=False)