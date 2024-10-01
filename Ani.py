import streamlit as st
import pickle
import numpy as np
import time

# Load the pre-trained model, scaler, and frequency mappings
model_path = r'C:\Users\sandy\Desktop\Project_realected_practice\Bank\Gradient_boost_model.pkl'
scale_path = r'C:\Users\sandy\Desktop\Project_realected_practice\Bank\Scale.pkl'
frequency_mappings_path = r'C:\Users\sandy\Desktop\Project_realected_practice\Bank\Fre_en.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scale_path, 'rb') as file:
    scaler = pickle.load(file)

with open(frequency_mappings_path, 'rb') as file:
    category_mappings = pickle.load(file)

# Function to format price into thousands, lakhs, or crores
def format_price(price):
    if price >= 1e7:
        return f"₹{price / 1e7:.2f} Cr"
    elif price >= 1e5:
        return f"₹{price / 1e5:.2f} Lakh"
    else:
        return f"₹{price / 1e3:.2f} Thousand"

# CSS for styling the app
css_code = """
<style>
body {
    background-image: url('file:///C:/Users/sandy/Downloads/vecteezy_ai-generated-polished-shiny-beautiful-black-car-on-dark_39617728.jpg'); /* Replace with your background image */
    background-size: cover;
    background-repeat: no-repeat;
    color: #ffffff; /* Text color */
    font-family: 'Arial', sans-serif; /* Font family */
}

h1, h3 {
    text-align: center;
    color: #ffcc00; /* Gold color for headings */
}

.sidebar .sidebar-content {
    background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent sidebar */
    color: #ffffff; /* Sidebar text color */
}

.stButton>button {
    background-color: #ff6600; /* Button color */
    color: white; /* Button text color */
}

.stButton>button:hover {
    background-color: #cc5200; /* Button hover color */
}

.spinner {
    color: #ffcc00; /* Spinner color */
}

.price {
    font-weight: bold;
    font-size: 24px;
}

.car-animation img {
    width: 300px;
    display: block;
    margin: auto;
}
</style>
"""

# Display the CSS in the Streamlit app
st.markdown(css_code, unsafe_allow_html=True)

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
    city = st.sidebar.selectbox("City", list(category_mappings['City'].keys()))
    body_type = st.sidebar.selectbox("Body Type", list(category_mappings['Body_type'].keys()))
    fuel_type = st.sidebar.selectbox("Fuel Type", list(category_mappings['Fuel_type'].keys()))
    transmission_type = st.sidebar.selectbox("Transmission Type", list(category_mappings['Transmission_Type'].keys()))
    manufactured_by = st.sidebar.selectbox("Manufactured By", list(category_mappings['Manufactured_By'].keys()))

    # Sliders for numerical features
    kilometers_driven = st.slider("Kilometers Driven", 0, 600000, 50000, step=500)
    previous_owners = st.slider("Previous Owners", 0, 8, 0)
    seats = st.slider("Seats", 2, 10, 5)
    mileage = st.slider("Mileage (kmpl)", 0.0, 50.0, 15.0)
    engine = st.slider("Engine (cc)", 50.0, 5000.0, 500.0)
    car_age = st.slider("Car Age (years)", 0, 40, 5)

    # Encoding categorical features based on the frequency mappings
    encoded_features = [
        category_mappings['Body_type'].get(body_type, 0),
        category_mappings['Manufactured_By'].get(manufactured_by, 0),
        category_mappings['City'].get(city, 0),
        category_mappings['Fuel_type'].get(fuel_type, 0),
        category_mappings['Transmission_Type'].get(transmission_type, 0),
    ]

    # Combine all features into a single input array
    features = np.array([
        kilometers_driven,
        previous_owners,
        seats,
        mileage,
        engine,
        car_age,
        *encoded_features
    ]).reshape(1, -1)

    # Predict button with animation effect
    if st.button("Predict Price"):
        # Display animation while predicting
        with st.spinner('Predicting...'):
            time.sleep(0.5)  # Simulate delay for prediction

        # Predict and format the price
        predicted_price = model.predict(scaler.transform(features))[0]
        formatted_price = format_price(predicted_price)

        # Display the estimated price
        st.markdown(f"<h3 class='price'>Estimated Vehicle Price is: {formatted_price}</h3>", unsafe_allow_html=True)

elif selected_page == "User Guide":
    st.header("User Guide")
    
    st.write("""

### How to Use the Vehicle Price Predictor App:

**Note:** If the sidebar is not visible, click the (>) shaped icon to make it appear.

1. **Select Categorical Features:** 
   Use the dropdown menus in the sidebar to select the city, body type, fuel type, transmission type, and manufacturer of the vehicle.
   
   - **Step 1:** Click the (>) shaped icon.
   - **Step 2:** Select one option based on your preference or what you are looking for. For example:
     - City → Chennai
     - Body Type → Hatchback
     - Fuel Type → Petrol
     - Transmission Type → Automatic
     - Manufactured By → Audi

2. **Adjust Numerical Features:** 
   Use the sliders to input the kilometers driven, number of previous owners, number of seats, mileage, engine size, and car age.
   
   - **Step 1:** Click the (>) shaped icon.
   - **Step 2:** Slide each slider based on your needs for each option.
   
   After completing these steps, click the **Predict Price** button.

3. **Predict Price:** 
   Click on the **'Predict Price'** button to estimate the vehicle's price.

4. **View Results:** 
   After clicking the button, the predicted price will be displayed prominently in bold and dark formatting.

**Note:** The price will be displayed in an appropriate unit (thousands, lakhs, crores) depending on its value.
""")
