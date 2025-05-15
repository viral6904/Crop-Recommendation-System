import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
def load_data():
    try:
        df = pd.read_csv('Crop_recommendation.csv')
    except FileNotFoundError:
        df = pd.DataFrame(np.random.rand(2100, 7), columns=['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall'])
        df['Crop'] = np.random.choice(['Wheat', 'Rice', 'Maize', 'Cotton', 'Barley'], 2100)
    return df

# Train models
def train_models(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {"accuracy": accuracy, "model": model}

    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    return models, results, best_model_name, results[best_model_name]['model'], X_test, y_test

# Crop Recommendation
def recommend_crop():
    try:
        user_data = [float(entry.get()) for entry in entries]
        prediction = best_model.predict(np.array(user_data).reshape(1, -1))[0]
        messagebox.showinfo("Recommendation", f"Best Model: {best_model_name}\nRecommended Crop: {prediction}")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for all fields.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Show Model Accuracy
def show_results():
    result_window = tk.Toplevel(root)
    result_window.title("Model Performance")
    result_window.configure(bg="#e0f7fa")
    ttk.Label(result_window, text="Model Performance Comparison", font=("Arial", 14, "bold"), background="#e0f7fa").pack(pady=10)
    for name, res in model_results.items():
        ttk.Label(result_window, text=f"{name}: Accuracy = {res['accuracy']:.2f}", background="#e0f7fa").pack()

# Evaluate User Input
def evaluate_entry(input_values, model, X_test, y_test):
    user_input = np.array(input_values).reshape(1, -1)
    predicted_crop = model.predict(user_input)[0]

    random_index = y_test.sample(1).index[0]
    actual_crop = y_test.loc[random_index]
    actual_features = X_test.loc[random_index].values

    expected_prediction = model.predict(actual_features.reshape(1, -1))[0]

    if predicted_crop == expected_prediction:
        if predicted_crop == actual_crop:
            return "True Positive"
        else:
            return "False Positive"
    else:
        if predicted_crop == actual_crop:
            return "False Negative"
        else:
            return "True Negative"

# Handle Prediction
def on_submit():
    try:
        values = [float(entry.get()) for entry in entries]
        result = evaluate_entry(values, best_model, X_test, y_test)
        messagebox.showinfo("Prediction Result", f"The model predicted: {best_model.predict(np.array(values).reshape(1, -1))[0]}\nEvaluation against a random test sample: {result}")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please enter numeric values for all fields.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Load and train models
data = load_data()
models, model_results, best_model_name, best_model, X_test, y_test = train_models(data)

# UI Setup
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# --- Root setup ---
root = tk.Tk()
root.title("🌱 Crop Recommendation System 🌞")
root.state('zoomed')
root.configure(bg="#ffeaa7")

# Load background image
try:
    bg_image = Image.open("backgroud1.jpg")
    bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
    bg_photo = ImageTk.PhotoImage(bg_image)

    canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight(), highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")
except Exception as e:
    print("Background image not loaded:", e)
    canvas = tk.Canvas(root, bg="#ffeaa7")
    canvas.pack(fill="both", expand=True)

# --- Style Setup ---
style = ttk.Style()
style.theme_use("clam")

style.configure("TLabel",
                background="#ffeaa7",  # Use yellow if fallback
                foreground="#2d3436",
                font=("Comic Sans MS", 11, "bold"))

style.configure("TEntry",
                font=("Comic Sans MS", 10),
                padding=5)

style.configure("Fun.TButton",
                font=("Comic Sans MS", 12, "bold"),
                padding=10,
                foreground="#2d3436",
                background="#fab1a0")

style.map("Fun.TButton",
          background=[("active", "#ff7675"), ("!active", "#fab1a0")],
          foreground=[("active", "#ffffff")])

# --- UI Components directly on canvas ---
widgets = []

def place_widget(widget, y_offset):
    canvas.create_window(root.winfo_screenwidth()//2, y_offset, window=widget)

title_label = ttk.Label(root, text="🌻 Enter Soil & Climate Parameters 🌈", font=("Comic Sans MS", 18, "bold"))
place_widget(title_label, 100)

fields = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"]
entries = []

y_offset = 160
for field in fields:
    frame = ttk.Frame(root)
    label = ttk.Label(frame, text=f"🎯 {field}:")
    entry = ttk.Entry(frame, width=25)
    label.pack(side=tk.LEFT, padx=10)
    entry.pack(side=tk.LEFT, padx=10)
    entries.append(entry)
    place_widget(frame, y_offset)
    y_offset += 50

recommend_button = ttk.Button(root, text="🚜 Recommend Crop", command=recommend_crop, style="Fun.TButton")
place_widget(recommend_button, y_offset + 20)

results_button = ttk.Button(root, text="📊 Show Model Performance", command=show_results, style="Fun.TButton")
place_widget(results_button, y_offset + 80)

evaluate_button = ttk.Button(root, text="🔍 Evaluate Prediction", command=on_submit, style="Fun.TButton")
place_widget(evaluate_button, y_offset + 140)

canvas.bg_photo = bg_photo if 'bg_photo' in locals() else None
root.mainloop()