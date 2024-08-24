import streamlit as st
import pickle
import numpy as np
import time

# Load the pre-trained model
model_path = 'Less_Gradient_boost_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

Scale_path = 'Scale.pkl'
with open(Scale_path, 'rb') as file:
    Scale = pickle.load(file)

# Define the categorical mappings
categorical_mappings = {
    "City": {"Bangalore": 0, "Chennai": 1, "Delhi": 2, "Hyderabad": 3, "Jaipur": 4, "Kolkata": 5},
    "Body_type": {"Convertibles": 0, "Coupe": 1, "Hatchback": 2, "Hybrids": 3, "MUV": 4, "Minivans": 5, "Pickup Trucks": 6, "SUV": 7, "Sedan": 8, "Wagon": 9},
    "Fuel_type": {"Cng": 0, "Diesel": 1, "Electric": 2, "Lpg": 3, "Petrol": 4},
    "Transmission_Type": {"Automatic": 0, "Manual": 1},
    "Manufactured_By": {"Audi": 0, "BMW": 1, "Chevrolet": 2, "Citroen": 3, "Datsun": 4, "Fiat": 5, "Ford": 6, "Hindustan Motors": 7, "Honda": 8, "Hyundai": 9, "Isuzu": 10, "Jaguar": 11, "Jeep": 12, "Kia": 13, "LandRover": 14, "Lexus": 15, "MG": 16, "Mahindra": 17, "Mahindra Renault": 18, "Mahindra Ssangyong": 19, 
    "Maruti": 20, "Mercedes-Benz": 21, "Mini": 22, "Mitsubishi": 23, "Nissan": 24, "Opel": 25, "Porsche": 26, 
    "Renault": 27, "Skoda": 28, "Tata": 29, "Toyota": 30, "Volkswagen": 31, "Volvo": 32
}}

# Function to format price into thousands, lakhs, or crores
def format_price(price):
    if price >= 1e7:
        return f"₹{price / 1e7:.2f} Cr"
    elif price >= 1e5:
        return f"₹{price / 1e5:.2f} Lakh"
    else:
        return f"₹{price / 1e3:.2f} Thousand"

# Title of the app
st.title("Vehicle Price Predictor")

# Add a sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Predict Price", "User Guide"]
selected_page = st.sidebar.selectbox("Choose a page", pages)

if selected_page == "Predict Price":
    # Sidebar for categorical inputs
    st.sidebar.header("Select Categorical Features")

    # Dropdowns for categorical features
    city = st.sidebar.selectbox("City", list(categorical_mappings['City'].keys()))
    body_type = st.sidebar.selectbox("Body Type", list(categorical_mappings['Body_type'].keys()))
    fuel_type = st.sidebar.selectbox("Fuel Type", list(categorical_mappings['Fuel_type'].keys()))
    transmission_type = st.sidebar.selectbox("Transmission Type", list(categorical_mappings['Transmission_Type'].keys()))
    manufactured_by = st.sidebar.selectbox("Manufactured By", list(categorical_mappings['Manufactured_By'].keys()))

    # Sliders for numerical features
    kilometers_driven = st.slider("Kilometers Driven", 0, 600000, 50000, step=500)
    previous_owners = st.slider("Previous Owners", 0, 8, 0)
    seats = st.slider("Seats", 2, 10, 5)
    mileage = st.slider("Mileage (kmpl)", 0.0, 50.0, 15.0)
    engine = st.slider("Engine (cc)", 50.0, 5000.0, 500.0)
    car_age = st.slider("Car Age (years)", 0, 40, 5)

    # Encoding categorical features based on the mapping
    encoded_features = [
        categorical_mappings['City'][city],
        categorical_mappings['Body_type'][body_type],
        categorical_mappings['Fuel_type'][fuel_type],
        categorical_mappings['Transmission_Type'][transmission_type],
        categorical_mappings['Manufactured_By'][manufactured_by],
    ]

    # Combine all features into a single input array
    features = np.array([
        *encoded_features,
        kilometers_driven,
        previous_owners,
        seats,
        mileage,
        engine,
        car_age
    ]).reshape(1, -1)
    
    # Predict button
    if st.button("Predict Price"):
        # Display animation while predicting
        with st.spinner('Predicting...'):
            time.sleep(0.5)  # Simulate delay for prediction

        # Predict and format the price
        predicted_price = model.predict(Scale.transform(features))[0]
        formatted_price = format_price(predicted_price)

        # Display the estimated price
        st.markdown(f"<h3 style='color:darkyellow; font-weight:bold;'>Estimated Vehicle Price is: {formatted_price}</h3>", unsafe_allow_html=True)

elif selected_page == "User Guide":
    st.header("User Guide")
    st.write("""
    ### How to Use the Vehicle Price Predictor App:
             
             * If incase the sidebar is not visible click the (>) shaped icon the sidebar will appear *
    
    1. **Select Categorical Features:** Use the dropdown menus in the sidebar to select the city, body type, fuel type, transmission type, and manufacturer of the vehicle.
             
            * Step 1 - Click the (V) shaped icon 
            * Step 2 - Slect the any one option based on your preference or what are you looking 
             for eg :
              City -> Chennai 
              Body Type --> Hatchback 
              Fuel Type --> Petrol 
              Transmission Type --> Automatic
              Manufactured By ---> Audi 
             Select as per your Preferences.

    2. **Adjust Numerical Features:** Use the sliders to input the kilometers driven, number of previous owners, number of seats, mileage, engine size, and car age.
             
              * Step 1 - Click the (V) shaped icon 
              * Step 2 - Slide the slider based on what you need in each options 
              * After done all these steps click Predict Price Button.
 
             
    3. **Predict Price:** Click on the 'Predict Price' button to estimate the vehicle's price.
             
    4. **View Results:** After clicking the button, the predicted price will be displayed in bold and dark formatting.
    
    The price will be displayed in an appropriate unit (thousands, lakhs, crores) depending on the value.
    """)


    
    st.write('**Demo Vedio:**')
    # Add the demo video
    video_path = "C:/Users/sandy/Downloads/Car_price_prediction_demo.mp4"
    st.video(video_path)
